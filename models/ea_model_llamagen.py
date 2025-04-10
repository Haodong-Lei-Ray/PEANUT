import copy
import json
import time
from typing import List, Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig
import numpy as np


from .kv_variants.modeling_llamagen_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .drafters.utils import *
from .drafters.kv_cache import initialize_past_key_values

from .drafters.cnets_llamagen import Model
from .configs.configs import EConfig

from .drafters.choices import *

import torch.nn.functional as F

def cfg_logit_process(combined_logits, cfg_scale=4.0):
    cond_logits, uncond_logits = torch.split(combined_logits, len(combined_logits) // 2, dim=0)
    logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    return logits

def calculate_tvd(tensor1, tensor2):
    tvd = 0.5 * torch.abs(tensor1 - tensor2)
    return tvd

def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        # logits = torch.ones_like(logits).to(logits.device)
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs

def pad_nested_list_left(nested_list):
    # Find the length of the longest sublist
    max_length = max(len(sublist) for sublist in nested_list)
    
    # Pad each sublist with 1s at the start (left padding only)
    padded_list = [[1] * (max_length - len(sublist)) + sublist for sublist in nested_list]
    
    return padded_list, max_length

class EaModel(nn.Module):

    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_config_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            ea_model_path
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        # self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path,use_fast=False)
        config = EConfig.from_pretrained(ea_model_config_path)
        with open(ea_model_config_path,"r") as f:
            con=json.loads(f.read())
        try:
            bias=con["bias"]
        except:
            bias=True
        self.ea_layer = Model(config,bias=bias,total_tokens=total_token,depth=depth,top_k=top_k,threshold=threshold)

        low_memory=False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device!=base_model.lm_head.weight.device:
            self.ea_layer.diff_device = True
            if not low_memory:
                self.ea_layer.headweight = base_model.lm_head.weight.clone().to(device)
            else:
                self.ea_layer.layer_device = device

        else:
            self.ea_layer.diff_device = False
        self.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()
        ea_model_dir = os.path.dirname(ea_model_config_path)
        nearest_latents_path = hf_hub_download(ea_model_path, "top_16383_indices.npy")
        self.nearest_latents = np.load(nearest_latents_path)

    # def get_tokenizer(self):
    #     """Get the tokenizer of the base model.

    #     Returns:
    #         Tokenizer: The tokenizer of the base model.
    #     """
    #     return self.tokenizer

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            total_token=59,
            depth=4,
            top_k=10,
            threshold=1.0,
            **kwargs,
    ):
        #assert Type=="LLaMA" or "Mixtral"
        Type=AutoConfig.from_pretrained(base_model_path).architectures[0]
        if Type=='LlamaForCausalLM':
            base_model = KVLlamaForCausalLM.from_pretrained(
                base_model_path, **kwargs
            )

        configpath=os.path.join(ea_model_path,"config.json")
        if not os.path.exists(configpath):
            configpath = hf_hub_download(ea_model_path, "config.json")
            # configpath = '/home/leihaodong/MM25/PEANUT/data/configs/llamagen_t2i_config.json'

        try:
            load_model_path=os.path.join(ea_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path=hf_hub_download(ea_model_path, "pytorch_model.bin")
            ea_layer_state_dict = torch.load(load_model_path,
                                             map_location=base_model.device)
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(ea_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
            ea_layer_state_dict = load_file(load_model_path)
        model = cls(
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            ea_model_path
        )



        if total_token==-1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans=[40,48,50,56,60]
            x=[1,1.05,1.07,1.1,1.13]
            times=[]

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token=cans[times.index(min(times))]
            model.ea_layer.total_tokens=total_token-1




        return model

    def forward(
            self,
            cond_idx=None,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                cond_idx=cond_idx,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    def pad_path(self, path: List[int], length: int, pad_value: int = -2) -> List[int]:
        """
        Pad the given path list with a specific value up to a specified length.

        Parameters:
        - path (list): The original list that needs padding.
        - length (int): The desired length of the padded list.
        - pad_value (optional, default=-2): The value to use for padding.

        Returns:
        - list: A new list based on the original path but padded to the desired length.

        Example:
        >>> pad_path([1,2,3], 5)
        [1, 2, 3, -2, -2]

        Note:
        If the given path is already longer than the specified length,
        then no padding occurs, and the original path is returned.
        """

        # Calculate the number of padding values needed by subtracting the length
        # of the path from the desired length.
        # Append the padding values to the original path and return the new list.
        return path + [pad_value] * (length - len(path))
        
    def generate_tree_buffers(self, tree_choices, device="cuda"):
        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        tree_len = len(sorted_tree_choices) + 1

        # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [self.pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                    dim=1)

        maxitem = retrieve_indices.max().item() + 5

        def custom_sort(lst):
            # sort_keys=[len(list)]
            sort_keys = []
            for i in range(len(lst)):
                sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
            return sort_keys

        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

        p_indices = torch.tensor(p_indices)
        p_indices_new = p_indices[retrieve_indices]
        p_indices_new = p_indices_new.tolist()

        b_indices = [[]] + b_indices
        b_indices_new = []
        for ib in range(retrieve_indices.shape[0]):
            iblist = []
            for jb in range(retrieve_indices.shape[1]):
                index = retrieve_indices[ib, jb]
                if index == -1:
                    iblist.append([])
                else:
                    b = b_indices[index]
                    if len(b) > 0:
                        bt = []
                        for bi in b:
                            bt.append(torch.where(tree_indices == bi)[0].item())
                        iblist.append(torch.tensor(bt, device=device))
                    else:
                        iblist.append(b)
            b_indices_new.append(iblist)

        # Aggregate the generated buffers into a dictionary
        tree_buffers = {
            "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
            "tree_indices": tree_indices,
            "tree_position_ids": tree_position_ids,
            "retrieve_indices": retrieve_indices,
        }

        # Move the tensors in the dictionary to the specified device
        tree_buffers = {
            k: v.clone().to(device)
            if isinstance(v, torch.Tensor)
            else torch.tensor(v, device=device)
            for k, v in tree_buffers.items()
        }
        tree_buffers["p_indices"] = p_indices_new
        tree_buffers["b_indices"] = b_indices_new
        return tree_buffers
    
    @torch.no_grad()
    def initialize_tree(self, cond_combined, past_key_values, logits_processor, cfg_scale, attention_mask = None):
        outputs, orig, hidden_states = self(
            cond_idx=cond_combined, past_key_values=past_key_values, output_orig=True, attention_mask=attention_mask
        )
        logits = cfg_logit_process(orig[:, -1], cfg_scale)

        if logits_processor is not None:
            logits = logits_processor(None, logits)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            token = torch.multinomial(probabilities, 1)
        else:
            token = torch.argmax(logits)
            token = token[None, None]
        token = torch.cat([token, token], dim=0)
        zero_padding = torch.zeros((token.shape[0], 120), dtype=torch.long, device=token.device)
        input_ids = torch.cat((zero_padding, token.to(cond_combined.device)), dim=1)
        draft_tokens, retrieve_indices,tree_mask,tree_position_ids = self.ea_layer.topK_genrate(hidden_states, input_ids, self.base_model.lm_head,logits_processor, cfg_scale)
        return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, orig, hidden_states, token
    
    @torch.no_grad()
    def initialize_tree_v1(self, cond_combined, tree_attn_mask, past_key_values, logits_processor, cfg_scale, attention_mask = None, tree_choices=mc_sim_7b_63):
        outputs, orig, hidden_states = self(
            cond_idx=cond_combined, past_key_values=past_key_values, output_orig=True, attention_mask=attention_mask
        )
        logits = cfg_logit_process(orig[:, -1], cfg_scale)
        if logits_processor is not None:
            logits = logits_processor(None, logits)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            token = torch.multinomial(probabilities, 1)
        else:
            token = torch.argmax(logits)
            token = token[None, None]
        token = torch.cat([token, token], dim=0)
        zero_padding = torch.zeros((token.shape[0], 120), dtype=torch.long, device=token.device)
        input_ids = torch.cat((zero_padding, token.to(cond_combined.device)), dim=1)
        self.ea_layer.init_tree_v1(tree_choices)
        tree_logits = self.ea_layer.topK_genrate_v1(hidden_states, input_ids, self.base_model.lm_head,logits_processor, cfg_scale)
        self.base_model.model.tree_mask = tree_attn_mask
        return tree_logits, logits, token

    @torch.no_grad()
    def evaluate_posterior_v1(
        self,
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
        cart_candidates_prob,
        op,
        p_indices,
        tree_candidates,
        b_indices,
        lantern=False,
        lantern_k=1000,
        lantern_delta=0.1,
    ) -> Tuple[torch.Tensor, int]:
        # Greedy decoding based on temperature value
        if logits_processor is None:
            device = logits.device
            batch_size, seq_len, vocab_size = logits.size()
            candidates_verify = candidates[:, 1:]  # Shape: (batch_size, seq_len)

            # Compute softmax probabilities over logits
            gtp = torch.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)

            # Get the token indices from candidates
            xi = candidates_verify  # Shape: (batch_size, seq_len)

            # Mask for positions where xi == -1
            valid_mask = (xi != -1).to(device)  # Shape: (batch_size, seq_len)

            # Adjust xi to have valid indices for indexing operations
            xi_valid = xi.clone()
            xi_valid[~valid_mask] = 0  # Replace invalid indices with 0 (or any valid index)

            # Gather probabilities of xi
            px = gtp.gather(dim=-1, index=xi_valid.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, seq_len)
            px = px * valid_mask  
            if isinstance(self.nearest_latents, np.ndarray):
                self.nearest_latents = torch.from_numpy(self.nearest_latents).to(device)
            if not lantern:
                # Greedy decoding
                top_tokens = torch.argmax(logits[:, :-1], dim=-1)  # Shape: (batch_size, seq_len)
                posterior_mask = (xi == top_tokens).int() * valid_mask
                candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
                accept_length = candidates_accept_length.max()
            else:
                # Adaptive decoding with nearest latent tokens
                search_space = lantern_k
                nearest_indices = self.nearest_latents[xi_valid]  # Shape: (batch_size, seq_len, k)
                nearest_indices = nearest_indices[:, :, :search_space]  # Limit search space

                # For invalid positions, set nearest_indices to zero
                nearest_indices[~valid_mask.unsqueeze(-1).expand_as(nearest_indices)] = 0

                # Get probabilities of nearest latent tokens
                nearest_probs = gtp.gather(dim=-1, index=nearest_indices)  # Shape: (batch_size, seq_len, search_space)
                nearest_probs = nearest_probs * valid_mask.unsqueeze(-1)  # Zero out invalid positions

                # Compute cumulative sum of nearest probabilities
                cumsum_nearest_probs = torch.cumsum(nearest_probs, dim=-1)  # Shape: (batch_size, seq_len, search_space)

                # Prepare target and approximate distributions
                px_expanded = px.unsqueeze(-1).repeat(1, 1, search_space)  # Shape: (batch_size, seq_len, search_space)
                approx_p = px_expanded + cumsum_nearest_probs  # Shape: (batch_size, seq_len, search_space)
                approx_p = approx_p * valid_mask.unsqueeze(-1)  # Zero out invalid positions

                # Concatenate distributions for TVD
                target_p = torch.cat([px_expanded, nearest_probs], dim=-1)  # Shape: (batch_size, seq_len, 2 * search_space)
                approx_p_full = torch.cat([approx_p, torch.zeros_like(nearest_probs)], dim=-1)

                # Zero out invalid positions in target and approximate distributions
                target_p = target_p * valid_mask.unsqueeze(-1).to(torch.float32)
                approx_p_full = approx_p_full * valid_mask.unsqueeze(-1).to(torch.float32)

                # Compute TVD
                tvd = calculate_tvd(target_p, approx_p_full)
            
                tvd = torch.nan_to_num(tvd, nan=0.0)
                tvd_px = tvd[:, :, :search_space]
                tvd_cumsum = torch.cumsum(tvd[:, :, search_space:], dim=-1)
                tvd = tvd_px + tvd_cumsum
                # For invalid positions, set tvd to a high value to avoid selecting them
                tvd[~valid_mask] = float('inf')

                # Determine indices where TVD exceeds threshold
                # Create a boolean mask where tvd does not exceed coeff_a
                if lantern_delta > 1.0:
                    tvd_not_exceeds = (tvd <= (lantern_delta - 1) * px.unsqueeze(-1))
                else:
                    tvd_not_exceeds = (tvd <= lantern_delta)

                # Get the size of the last dimension
                dim_size = tvd.shape[-1]

                # Create indices for the last dimension
                indices = torch.arange(dim_size).unsqueeze(0).unsqueeze(0).to(tvd.device)
                indices = indices.expand(tvd.shape[0], tvd.shape[1], dim_size)

                # Use the mask to select valid indices, set invalid positions to -1
                masked_indices = torch.where(tvd_not_exceeds, indices, torch.full_like(indices, -1))

                # Find the maximum valid index for each (batch_size, seq_len)
                indices = masked_indices.max(dim=-1)[0]

                # Update probabilities based on indices
                idx_mask = (indices >= 0)
                idx_values = indices * idx_mask
                idx_values = idx_values.unsqueeze(-1)

                # Handle positions where idx_values == -1
                px_adjusted = torch.where(
                    idx_mask,
                    approx_p.gather(dim=-1, index=idx_values).squeeze(-1),
                    px
                )
                px_adjusted = px_adjusted * valid_mask  # Zero out invalid positions

                # Update gtp with adjusted probabilities
                gtp.scatter_(dim=-1, index=xi_valid.unsqueeze(-1), src=px_adjusted.unsqueeze(-1))

                # Compute posterior mask
                top_tokens = torch.argmax(gtp, dim=-1)[:, :-1]  # Adjusted to match xi
                posterior_mask = (xi == top_tokens).int() * valid_mask
                candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
                accept_length = candidates_accept_length.max()

            # Choose the best candidate
            if accept_length == 0:
                best_candidate = torch.tensor(0, dtype=torch.long, device=device)
            else:
                best_candidate = torch.argmax(candidates_accept_length).to(torch.long)

            return best_candidate, accept_length, logits[best_candidate, accept_length]

        else:
            cart_candidates_prob = cart_candidates_prob.to(logits.device)
            accept_length = 1
            accept_cand = candidates[0][:1]
            best_candidate = 0
            for i in range(1, candidates.shape[1]):
                if i != accept_length:
                    break
                adjustflag = False
                is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
                fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
                gt_logits = logits[fi, i - 1][None]
                gt_logits = logits_processor(None, gt_logits)[0]
                gtp = torch.softmax(gt_logits, dim=0)
                candidates_set = []
                for j in range(candidates.shape[0]):
                    if is_eq[j]:
                        x = candidates[j, i]
                        xi = x.item()
                        if xi in candidates_set or xi == -1:
                            continue
                        candidates_set.append(xi)
                        r = random.random()
                        px = gtp[xi]
                        if lantern:
                            nearest_probs = gtp[self.nearest_latents[xi, :lantern_k]].reshape(lantern_k, 1)
                            cumsum_nearest_probs = torch.cumsum(nearest_probs, dim=0)
                            if lantern_delta > 1.0:
                                indices = (cumsum_nearest_probs <= (lantern_delta - 1) * px).nonzero(as_tuple=True)[0]
                            else:
                                indices = (cumsum_nearest_probs <= lantern_delta).nonzero(as_tuple=True)[0]
                            if indices.numel() == 0:
                                indices = -1
                            else:
                                indices = indices[-1]
                            if indices == -1:
                                px = px
                            else:
                                px = px + cumsum_nearest_probs[indices]
                        qx = cart_candidates_prob[j, i]
                        if qx <= 0:
                            continue
                        acp = px / qx
                        if r <= acp:
                            accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                            accept_length += 1
                            best_candidate = j
                            break
                        else:
                            q = op[i - 1][p_indices[j][i]].clone()
                            b = b_indices[j][i]
                            if len(b) > 0:
                                mask = tree_candidates[0][b]
                                q[mask] = 0
                                q = q / q.sum()
                            if lantern:
                                if (indices != -1):
                                    q[self.nearest_latents[xi, :lantern_k + 1]] = 0
                            gtp = gtp - q
                            gtp[gtp < 0] = 0

                            if gtp.sum() == 0:
                                gtp = torch.ones_like(gtp)

                            gtp = gtp / gtp.sum()
                            adjustflag = True
            if adjustflag and accept_length != candidates.shape[1]:
                sample_p = gtp
            else:
                gt_logits = logits[best_candidate, accept_length - 1][None]
                gt_logits = logits_processor(None, gt_logits)[0]
                sample_p = torch.softmax(gt_logits, dim=0)
            return torch.tensor(best_candidate), accept_length - 1, sample_p
    

    def reset_tree_mode(self):
        self.base_model.model.tree_mode = True
        self.base_model.model.tree_mask = None
    
    def generate_candidates(self, tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
        sample_token = sample_token.to(tree_indices.device)

        candidates_logit = sample_token[0]

        candidates_tree_logits = tree_logits[0]

        candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

        tree_candidates = candidates[tree_indices]

        tree_candidates_ext = torch.cat(
            [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

        cart_candidates = tree_candidates_ext[retrieve_indices]

        if logits_processor is not None:
            candidates_tree_prob = tree_logits[1]
            candidates_prob = torch.cat(
                [torch.ones(1, device=candidates_tree_prob.device, dtype=torch.float32), candidates_tree_prob.view(-1)],
                dim=-1)

            tree_candidates_prob = candidates_prob[tree_indices]
            tree_candidates_prob_ext = torch.cat(
                [tree_candidates_prob, torch.ones((1), dtype=torch.float32, device=tree_candidates_prob.device)], dim=0)
            cart_candidates_prob = tree_candidates_prob_ext[retrieve_indices]
        else:
            cart_candidates_prob = None
        # Unsqueeze the tree candidates for dimension consistency.
        tree_candidates = tree_candidates.unsqueeze(0)
        return cart_candidates, cart_candidates_prob, tree_candidates
    
    
    def evaluate_posterior(self, logits, candidates, logits_processor=None, lantern=False, lantern_k=1000, lantern_delta=0.1):
        if logits_processor is not None:
            accept_length = 1
            accept_cand = candidates[0][:1]
            best_candidate = 0

            # for-loop over levels
            for i in range(1, candidates.shape[1]):
                if i != accept_length:
                    break
                
                adjustflag = False
                is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
                fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
                
                gt_logits = logits[fi, i-1][None]
                gt_logits = logits_processor(None, gt_logits)[0]
                gtp = torch.softmax(gt_logits, dim=0)
                
                candidates_set = []

                # for-loop within a level
                for j in range(candidates.shape[0]):
                    if is_eq[j]:
                        x = candidates[j, i]
                        xi = x.item()

                        if xi in candidates_set or xi == -1:
                            continue
                        
                        candidates_set.append(xi)

                        r = random.random()
                        px = gtp[xi]
                        if lantern:
                            nearest_probs = gtp[self.nearest_latents[xi, :lantern_k]].reshape(lantern_k, 1)
                            cumsum_nearest_probs = torch.cumsum(nearest_probs, dim=0)

                            if lantern_delta > 1.0:
                                indices = (cumsum_nearest_probs <= (lantern_delta - 1) * px).nonzero(as_tuple=True)[0]
                            else:
                                indices = (cumsum_nearest_probs <= lantern_delta).nonzero(as_tuple=True)[0]
                            if indices.numel() == 0:
                                indices = -1
                            else:
                                indices = indices[-1]
                            if indices == -1:
                                px = px
                            else:
                                px = px + cumsum_nearest_probs[indices]
                        
                        qx = 1.0
                        acp = px / qx

                        if r <= acp:
                            accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                            accept_length += 1
                            best_candidate = j
                            break
                        else:
                            gtp[xi] = 0
                            
                            if lantern:
                                if (indices != -1):
                                    gtp[self.nearest_latents[xi, :lantern_k+1]] = 0
                            
                            if gtp.sum() == 0:
                                gtp = torch.ones_like(gtp)
                            
                            gtp /= gtp.sum()
                            adjustflag = True

            if adjustflag and accept_length != candidates.shape[1]:
                sample_p = gtp
            else:
                gt_logits = logits[best_candidate, accept_length-1][None]
                gt_logits = logits_processor(None, gt_logits)[0]
                sample_p = torch.softmax(gt_logits, dim=0)
            return torch.tensor(best_candidate), accept_length-1, sample_p
        
        else:
            device = logits.device
            batch_size, seq_len, vocab_size = logits.size()
            candidates_verify = candidates[:, 1:]  # Shape: (batch_size, seq_len)

            # Compute softmax probabilities over logits
            gtp = torch.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)

            # Get the token indices from candidates
            xi = candidates_verify  # Shape: (batch_size, seq_len)

            # Mask for positions where xi == -1
            valid_mask = (xi != -1).to(device)  # Shape: (batch_size, seq_len)

            # Adjust xi to have valid indices for indexing operations
            xi_valid = xi.clone()
            xi_valid[~valid_mask] = 0  # Replace invalid indices with 0 (or any valid index)

            # Gather probabilities of xi
            px = gtp.gather(dim=-1, index=xi_valid.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, seq_len)
            px = px * valid_mask  
            if isinstance(self.nearest_latents, np.ndarray):
                self.nearest_latents = torch.from_numpy(self.nearest_latents).to(device)
            if not lantern:
                # Greedy decoding
                top_tokens = torch.argmax(logits[:, :-1], dim=-1)  # Shape: (batch_size, seq_len)
                posterior_mask = (xi == top_tokens).int() * valid_mask
                candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
                accept_length = candidates_accept_length.max()
            else:
                # Adaptive decoding with nearest latent tokens
                search_space = lantern_k
                nearest_indices = self.nearest_latents[xi_valid]  # Shape: (batch_size, seq_len, k)
                nearest_indices = nearest_indices[:, :, :search_space]  # Limit search space

                # For invalid positions, set nearest_indices to zero
                nearest_indices[~valid_mask.unsqueeze(-1).expand_as(nearest_indices)] = 0

                # Get probabilities of nearest latent tokens
                nearest_probs = gtp.gather(dim=-1, index=nearest_indices)  # Shape: (batch_size, seq_len, search_space)
                nearest_probs = nearest_probs * valid_mask.unsqueeze(-1)  # Zero out invalid positions

                # Compute cumulative sum of nearest probabilities
                cumsum_nearest_probs = torch.cumsum(nearest_probs, dim=-1)  # Shape: (batch_size, seq_len, search_space)

                # Prepare target and approximate distributions
                px_expanded = px.unsqueeze(-1).repeat(1, 1, search_space)  # Shape: (batch_size, seq_len, search_space)
                approx_p = px_expanded + cumsum_nearest_probs  # Shape: (batch_size, seq_len, search_space)
                approx_p = approx_p * valid_mask.unsqueeze(-1)  # Zero out invalid positions

                # Concatenate distributions for TVD
                target_p = torch.cat([px_expanded, nearest_probs], dim=-1)  # Shape: (batch_size, seq_len, 2 * search_space)
                approx_p_full = torch.cat([approx_p, torch.zeros_like(nearest_probs)], dim=-1)

                # Zero out invalid positions in target and approximate distributions
                target_p = target_p * valid_mask.unsqueeze(-1).to(torch.float32)
                approx_p_full = approx_p_full * valid_mask.unsqueeze(-1).to(torch.float32)

                # Compute TVD
                tvd = calculate_tvd(target_p, approx_p_full)
            
                tvd = torch.nan_to_num(tvd, nan=0.0)
                tvd_px = tvd[:, :, :search_space]
                tvd_cumsum = torch.cumsum(tvd[:, :, search_space:], dim=-1)
                tvd = tvd_px + tvd_cumsum
                # For invalid positions, set tvd to a high value to avoid selecting them
                tvd[~valid_mask] = float('inf')

                # Determine indices where TVD exceeds threshold
                # Create a boolean mask where tvd does not exceed coeff_a
                if lantern_delta > 1.0:
                    tvd_not_exceeds = (tvd <= (lantern_delta - 1) * px.unsqueeze(-1))
                else:
                    tvd_not_exceeds = (tvd <= lantern_delta)

                # Get the size of the last dimension
                dim_size = tvd.shape[-1]

                # Create indices for the last dimension
                indices = torch.arange(dim_size).unsqueeze(0).unsqueeze(0).to(tvd.device)
                indices = indices.expand(tvd.shape[0], tvd.shape[1], dim_size)

                # Use the mask to select valid indices, set invalid positions to -1
                masked_indices = torch.where(tvd_not_exceeds, indices, torch.full_like(indices, -1))

                # Find the maximum valid index for each (batch_size, seq_len)
                indices = masked_indices.max(dim=-1)[0]

                # Update probabilities based on indices
                idx_mask = (indices >= 0)
                idx_values = indices * idx_mask
                idx_values = idx_values.unsqueeze(-1)

                # Handle positions where idx_values == -1
                px_adjusted = torch.where(
                    idx_mask,
                    approx_p.gather(dim=-1, index=idx_values).squeeze(-1),
                    px
                )
                px_adjusted = px_adjusted * valid_mask  # Zero out invalid positions

                # Update gtp with adjusted probabilities
                gtp.scatter_(dim=-1, index=xi_valid.unsqueeze(-1), src=px_adjusted.unsqueeze(-1))

                # Compute posterior mask
                top_tokens = torch.argmax(gtp, dim=-1)[:, :-1]  # Adjusted to match xi
                posterior_mask = (xi == top_tokens).int() * valid_mask
                candidates_accept_length = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
                accept_length = candidates_accept_length.max()

            # Choose the best candidate
            if accept_length == 0:
                best_candidate = torch.tensor(0, dtype=torch.long, device=device)
            else:
                best_candidate = torch.argmax(candidates_accept_length).to(torch.long)

            return best_candidate, accept_length, logits[best_candidate, accept_length]
        
    @torch.no_grad()
    def tree_decoding(
        self,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
        cfg_scale,
        attention_mask = None,
): 
        position_ids = tree_position_ids + input_ids.shape[1]
        if attention_mask is not None:
            remaining_length = input_ids.shape[1] + tree_candidates.shape[1] - attention_mask.shape[1]
            one_padding = torch.ones((attention_mask.shape[0], remaining_length), dtype=torch.long, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, one_padding], dim=1)
        outputs, tree_logits, hidden_state = self(
            input_ids=tree_candidates,
            output_orig=True,
            past_key_values=past_key_values,
            position_ids=position_ids,
            attention_mask=attention_mask
        )
        tree_logits = cfg_logit_process(tree_logits, cfg_scale)
        logits = tree_logits[0, retrieve_indices]
        return logits, hidden_state, outputs

    @torch.no_grad()
    def update_inference_inputs(
        self,
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        hidden_state_new,
        sample_p,
        cfg_scale,
        static_tree=False
    ):
        prev_input_len = input_ids.shape[1]

        select_indices = (
                retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
        )

        input_ids = torch.cat(
            [input_ids, candidates[None, best_candidate, : accept_length + 1]], dim=-1
        )
        # Update the past key values based on the selected tokens
        # Source tensor that contains relevant past information based on the selected candidate
        for past_key_values_data in past_key_values_data_list:
            tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
            # Destination tensor where the relevant past information will be stored
            dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
            # Copy relevant past information from the source to the destination
            dst.copy_(tgt, non_blocking=True)

        # Update the current length tensor (currently only support batch size is 1)
        current_length_data.fill_(prev_input_len + tgt.shape[-2])

        retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
        accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
        # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
        # token=token[None,None]
        prob = sample_p
        if logits_processor is not None:
            token = torch.multinomial(prob, 1)
            token = token[None]
        else:
            token = torch.argmax(prob)
            token = token[None, None]
        # hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
        ea_input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1).repeat(2, 1)
        
        if static_tree:
            tree_logits = self.ea_layer.topK_genrate_v1(accept_hidden_state_new,
                                                        input_ids=ea_input_ids,
                                                        head=self.base_model.lm_head,logits_processor=logits_processor,
                                                        cfg_scale=cfg_scale)
            new_token += accept_length + 1
            return input_ids, tree_logits, new_token, None, token
        else:
            draft_tokens, retrieve_indices,tree_mask,tree_position_ids = self.ea_layer.topK_genrate(accept_hidden_state_new,
                                                    input_ids=ea_input_ids,
                                                    head=self.base_model.lm_head,logits_processor=logits_processor,
                                                    cfg_scale=cfg_scale)
            new_token += accept_length + 1
            return input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, None, token

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        cfg: Optional[float] = None,
        lantern: Optional[bool] = None,
        lantern_k: Optional[int] = None,
        lantern_delta: Optional[float] = None,
        static_tree: Optional[bool] = None,
        tree_choices: Optional[List[List[int]]] = naive_extend_57,
        **model_kwargs,
    ):
        accept_length_list = []
        caption_embs, emb_masks = self.base_model.t5_model.get_text_embeddings(prompt)
        new_emb_masks = torch.flip(emb_masks, dims=[-1])
        new_caption_embs = []
        for idx, (caption_emb, emb_mask) in enumerate(zip(caption_embs, emb_masks)):
            valid_num = int(emb_mask.sum().item())
            # print(f'  prompt {idx} token len: {valid_num}')
            new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
            new_caption_embs.append(new_caption_emb)
        new_caption_embs = torch.stack(new_caption_embs)
        
        c_indices = new_caption_embs * new_emb_masks[:,:, None]
        c_emb_masks = new_emb_masks
        st = time.time()
        if hasattr(self.base_model, "past_key_values"):
            past_key_values = self.base_model.past_key_values
            past_key_values_data = self.base_model.past_key_values_data
            current_length_data = self.base_model.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, 2)
            self.base_model.past_key_values = past_key_values
            self.base_model.past_key_values_data = past_key_values_data
            self.base_model.current_length_data = current_length_data
        if cfg is not None:
            cond_null = torch.zeros_like(c_indices, device=c_indices.device) + self.base_model.model.cls_embedding.uncond_embedding.to(c_indices.device)
            cond_combined = torch.cat([c_indices, cond_null])
        else:
            cond_combined = c_indices
        T = cond_combined.shape[1]
        max_batch_size = c_indices.shape[0]
        if c_emb_masks is not None:
            assert c_emb_masks.shape[0] == max_batch_size
            assert c_emb_masks.shape[1] == T
            if cfg is not None:
                attention_mask = torch.cat([c_emb_masks, c_emb_masks])
            else:
                attention_mask = c_emb_masks
        cond_combined = cond_combined.to(self.base_model.dtype)
        padding = (torch.zeros(1,1,dtype=torch.long)-1).to(cond_combined.device)
        self.ea_layer.reset_kv()
        
        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_k=top_k, top_p=top_p)
        else:
            logits_processor = None
        
        st = time.time()
        if static_tree:
            if hasattr(self, "tree_choices") and self.tree_choices == tree_choices:
                tree_buffers = self.tree_buffers
            else:
                tree_buffers = self.generate_tree_buffers(
                    tree_choices, device=self.base_model.model.layers[-1].self_attn.q_proj.weight.device
                )
                tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(self.base_model.lm_head.weight.device)
            self.tree_buffers = tree_buffers
            self.tree_choices = tree_choices

        if hasattr(self.base_model, "past_key_values"):
            past_key_values = self.base_model.past_key_values
            past_key_values_data = self.base_model.past_key_values_data
            current_length_data = self.base_model.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, 2)
            self.base_model.past_key_values = past_key_values
            self.base_model.past_key_values_data = past_key_values_data
            self.base_model.current_length_data = current_length_data
            
        self.reset_tree_mode()

        if static_tree:
            tree_logits, logits, sample_token = self.initialize_tree_v1(
                cond_combined, tree_buffers['tree_attn_mask'], past_key_values, logits_processor, cfg, attention_mask, tree_choices
            )
        else:
            draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token = self.initialize_tree(
                cond_combined, past_key_values, logits_processor, cfg, attention_mask
            )

        max_steps = max_length
        input_ids = torch.zeros((max_batch_size, 120), dtype=torch.long).to(cond_combined.device)
        new_token=0
        for idx in range(max_steps):
            if static_tree:
                candidates, cart_candidates_prob, tree_candidates = self.generate_candidates(
                    tree_logits, tree_buffers["tree_indices"], tree_buffers["retrieve_indices"], sample_token, logits_processor
                )
                tree_candidates = torch.cat([tree_candidates, tree_candidates]).to(self.base_model.device)
                logits, hidden_state_new, outputs = self.tree_decoding(
                    tree_candidates, past_key_values, tree_buffers["tree_position_ids"], input_ids, tree_buffers["retrieve_indices_head"], cfg, attention_mask
                )
                best_candidate, accept_length, sample_p = self.evaluate_posterior_v1(
                    logits, candidates, logits_processor, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"], tree_candidates, tree_buffers["b_indices"], lantern, lantern_k, lantern_delta
                )
                input_ids, tree_logits, new_token, hidden_state, sample_token= self.update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    tree_buffers["retrieve_indices_head"],
                    logits_processor,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    hidden_state_new,
                    sample_p,
                    cfg,
                    static_tree=static_tree
                )

            else:
                self.base_model.model.tree_mask = tree_mask
                
                tree_draft_tokens = torch.cat([draft_tokens, draft_tokens]).to(self.base_model.device)
                
                logits, hidden_state_new, outputs = self.tree_decoding(
                    tree_draft_tokens, past_key_values, tree_position_ids, input_ids, retrieve_indices, cfg, attention_mask
                )
                draft_tokens = torch.cat((draft_tokens, padding), dim=1)
                candidates = draft_tokens[0, retrieve_indices]

                best_candidate, accept_length, sample_p = self.evaluate_posterior(logits, candidates,  logits_processor, lantern=lantern, lantern_k=lantern_k, lantern_delta=lantern_delta)
                
                input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = self.update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    retrieve_indices,
                    logits_processor,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    hidden_state_new,
                    sample_p,
                    cfg,
                )
            if torch.is_tensor(accept_length):
                accept_length_list.append(accept_length.item()+1)
            else:
                accept_length_list.append(accept_length+1)
            if new_token > max_length:
                break
        return input_ids[:, 120:120+max_length], sum(accept_length_list)/len(accept_length_list), time.time()-st
    
    @torch.no_grad()
    def decode_ids(self, ids):
        return self.base_model.decode_ids(ids)