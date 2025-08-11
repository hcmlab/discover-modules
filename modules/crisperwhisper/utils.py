import numpy as np
import torch
import torch.nn as nn

def _adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
        """
        Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.
        
        TODO: Handle overlapping timestamps that can occur after pause distribution adjustments
        """

        adjusted_chunks = pipeline_output["chunks"].copy()
        
        # Calculate average duration for None timestamp approximation
        durations = []
        for chunk in adjusted_chunks:
            start, end = chunk["timestamp"]
            if start is not None and end is not None:
                durations.append(end - start)
        avg_duration = sum(durations) / len(durations) if durations else 0.1

        for i in range(len(adjusted_chunks) - 1):
            current_chunk = adjusted_chunks[i]
            next_chunk = adjusted_chunks[i + 1]

            current_start, current_end = current_chunk["timestamp"]
            next_start, next_end = next_chunk["timestamp"]
            
            # Fix for None timestamp in next_end using average duration
            if next_end is None:
                next_end = next_start + avg_duration
            
            pause_duration = next_start - current_end

            if pause_duration > 0:
                if pause_duration > split_threshold:
                    distribute = split_threshold / 2
                else:
                    distribute = pause_duration / 2

                # Adjust current chunk end time
                adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

                # Adjust next chunk start time
                adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
        pipeline_output["chunks"] = adjusted_chunks

        return pipeline_output

def _median_filter(inputs: torch.Tensor, filter_width: int) -> torch.Tensor:
    """
    Applies a median filter of width `filter_width` along the last dimension of the input.

    The `inputs` tensor is assumed to be 3- or 4-dimensional.
    """
    if filter_width <= 0 or filter_width % 2 != 1:
        raise ValueError("`filter_width` should be an odd number")

    pad_width = filter_width // 2
    if inputs.shape[-1] <= pad_width:
        return inputs

    # Pad the left and right edges.
    inputs = nn.functional.pad(inputs, (pad_width, pad_width, 0, 0), mode="reflect")

    # sort() is faster than torch.median (https://github.com/pytorch/pytorch/issues/51450)
    result = inputs.unfold(-1, filter_width, 1).sort()[0][..., pad_width]
    return result

def _dynamic_time_warping2(matrix: np.ndarray, allow_vertical_moves: bool = True):
    """
    Measures similarity between two temporal sequences: the input audio and the output tokens. Used to generate
    token-level timestamps.
    """
    output_length, input_length = matrix.shape
    cost = np.ones((output_length + 1, input_length + 1), dtype=np.float32) * np.inf
    trace = -np.ones((output_length + 1, input_length + 1), dtype=np.float32)

    cost[0, 0] = 0
    for j in range(1, input_length + 1):
        for i in range(1, output_length + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j] if allow_vertical_moves else np.inf
            c2 = cost[i, j - 1]

            if c0 <= c1 and c0 <= c2:
                c, t = c0, 0
            elif c1 <= c0 and c1 <= c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = matrix[i - 1, j - 1] + c
            trace[i, j] = t

    # backtrace
    i = output_length
    j = input_length
    trace[0, :] = 2
    trace[:, 0] = 1

    text_indices = []
    time_indices = []
    while i > 0 or j > 0:
        text_indices.append(i - 1)
        time_indices.append(j - 1)
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        elif trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1 and allow_vertical_moves:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            # Handle unexpected cases by moving diagonally
            i -= 1
            j -= 1

    text_indices = np.array(text_indices)[::-1]
    time_indices = np.array(time_indices)[::-1]
    return text_indices, time_indices

def _patched_extract_token_timestamps(self, generate_outputs, alignment_heads, time_precision=0.02, num_frames=None):
        """
        Calculates token-level timestamps using the encoder-decoder cross-attentions and dynamic time-warping (DTW) to
        map each output token to a position in the input audio. If `num_frames` is specified, the encoder-decoder
        cross-attentions will be cropped before applying DTW.

        Returns:
            tensor containing the timestamps in seconds for each predicted token
        """
        # Create a list with `decoder_layers` elements, each a tensor of shape
        # (batch size, attention_heads, output length, input length).
        cross_attentions = []
        for i in range(self.config.decoder_layers):
            cross_attentions.append(torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2))

        # Select specific cross-attention layers and heads. This is a tensor
        # of shape (batch size, num selected, output length, input length).
        weights = torch.stack([cross_attentions[l][:, h] for l, h in alignment_heads])
        weights = weights.permute([1, 0, 2, 3])

        if "beam_indices" in generate_outputs:
            # If beam search has been used, the output sequences may have been generated for more timesteps than their sequence_lengths
            # since the beam search strategy chooses the most probable sequences at the end of the search.
            # In that case, the cross_attentions weights are too long and we have to make sure that they have the right output_length
            weight_length = (generate_outputs.beam_indices != -1).sum(-1).max()
            weights = weights[:, :, :weight_length]

            # If beam index is still -1, it means that the associated token id is EOS
            # We need to replace the index with 0 since index_select gives an error if any of the indexes is -1.
            beam_indices = generate_outputs.beam_indices[:, :weight_length]
            beam_indices = beam_indices.masked_fill(beam_indices == -1, 0)

            # Select the cross attention from the right beam for each output sequences
            weights = torch.stack(
                [
                    torch.index_select(weights[:, :, i, :], dim=0, index=beam_indices[:, i])
                    for i in range(beam_indices.shape[1])
                ],
                dim=2,
            )

        timestamps = torch.zeros_like(generate_outputs.sequences, dtype=torch.float32)
        batch_size = timestamps.shape[0]

        if num_frames is not None:
            # two cases:
            # 1. num_frames is the same for each sample -> compute the DTW matrix for each sample in parallel
            # 2. num_frames is different, compute the DTW matrix for each sample sequentially

            # we're using np.unique because num_frames can be int/list/tuple
            if len(np.unique(num_frames)) == 1:
                # if num_frames is the same, no need to recompute matrix, std and mean for each element of the batch
                num_frames = num_frames if isinstance(num_frames, int) else num_frames[0]

                weights = weights[..., : num_frames // 2]
            else:
                # num_frames is of shape (batch_size,) whereas batch_size is truely batch_size*num_return_sequences
                repeat_time = batch_size if isinstance(num_frames, int) else batch_size // len(num_frames)
                num_frames = np.repeat(num_frames, repeat_time)

        if num_frames is None or isinstance(num_frames, int):
            # Normalize and smoothen the weights.
            if self.generation_config.legacy:
                std = torch.std(weights, dim=-2, keepdim=True, unbiased=False)
                mean = torch.mean(weights, dim=-2, keepdim=True)
                weights = (weights - mean) / std
            weights = _median_filter(weights, self.generation_config.median_filter_width)

            # Average the different cross-attention heads.
            weights = weights.mean(dim=1)

        # Perform dynamic time warping on each element of the batch.
        for batch_idx in range(batch_size):
            non_special_tokens_indices = []
            special_tokens_indices = []
            for index_, token_id in enumerate(generate_outputs.sequences[batch_idx, 1:]):
                if self.generation_config.token_ids_to_ignore_for_dtw:
                    if token_id not in self.generation_config.token_ids_to_ignore_for_dtw and token_id < self.model.config.eos_token_id:
                        non_special_tokens_indices.append(index_)
                    else:
                        special_tokens_indices.append(index_)
            if num_frames is not None and isinstance(num_frames, (tuple, list, np.ndarray)):
                matrix = weights[batch_idx, :, non_special_tokens_indices, :num_frames[batch_idx] // 2]

                # Normalize and smoothen the weights.
                if self.generation_config.legacy:
                    std = torch.std(matrix, dim=-2, keepdim=True, unbiased=False)
                    mean = torch.mean(matrix, dim=-2, keepdim=True)
                    matrix = (matrix - mean) / std
                matrix = _median_filter(matrix, self.generation_config.median_filter_width)

                # Average the different cross-attention heads.
                matrix = matrix.mean(dim=0)
            else:
                matrix = weights[batch_idx, non_special_tokens_indices, :]

            text_indices, time_indices = _dynamic_time_warping2(-matrix.cpu().double().numpy(), allow_vertical_moves=False)
            jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
            jump_times = list((time_indices[jumps] * time_precision).round(4))
            for index_ in special_tokens_indices:
                # since jump_times is only extended in the next step, take index_ and not (index_ + 1)
                next_element = jump_times[index_] if index_ < len(jump_times) else round((time_indices[-1] * time_precision), 4)
                jump_times.insert(index_, next_element)
            if timestamps.shape[-1] == len(jump_times):
                timestamps[batch_idx, 0:] = torch.tensor(jump_times)
            else:
                timestamps[batch_idx, 1:] = torch.tensor(jump_times)

        return timestamps