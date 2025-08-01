import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if int(os.environ.get("RANK", 0)) != 0:
    os.environ["TQDM_DISABLE"] = "1"

import argparse
import time

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler,
)
from diffusers.utils import export_to_video
from termcolor import colored

from dax.adapters.wan import optimize_pipe
from dax.cache.teacache import reset_cache
from dax.parallel import dist as para_dist
from dax.parallel.dist.coordinator import destroy_process_group
from dax.parallel.dist.log_rank import log_rank0
from dax.parallel.initialize import is_initialized
from dax.utils import seed_everything


@torch.no_grad
def main():
    parser = argparse.ArgumentParser(
        description="Generate video from text prompt using Wan-Diffuser"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-14B-Diffusers",
        help="Model ID to use for generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative text prompt to avoid certain features",
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Height of the generated video"
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Width of the generated video"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames in the generated video",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps in the generated video",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output.mp4",
        help="Output video file name",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for generation"
    )
    parser.add_argument(
        "--sequence_parallel",
        action="store_true",
        help="Whether to enable sequence parallel",
    )
    parser.add_argument(
        "--int8_linear",
        action="store_true",
        help="Whether to enable int8 linear quantization",
    )
    parser.add_argument(
        "--overlap_comm",
        action="store_true",
        help="Whether to enable overlap communication",
    )
    parser.add_argument(
        "--enable_teacache",
        action="store_true",
        help="Whether to use teacache",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether to use compile",
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
    model_id = args.model_id
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    )
    flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=flow_shift,
    )
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler

    if args.enable_teacache:
        kwargs = {
            "cache_strategy": "teacache",
            "teacache_thresh": 0.1,
            "num_inference_steps": args.num_inference_steps,
        }
    else:
        kwargs = {}

    optimize_pipe(
        pipe,
        sequence_parallel=args.sequence_parallel,
        compile=args.compile,
        int8_linear=args.int8_linear,
        overlap_comm=args.overlap_comm,
        **kwargs,
    )

    if is_initialized():
        device = torch.device(f"cuda:{para_dist.get_local_rank()}")
    else:
        device = torch.device("cuda")
    pipe.to(device)

    if args.prompt is None:
        log_rank0(colored("Using default prompt", "red"))
        args.prompt = """
        The camera rushes from far to near in a low-angle shot, 
        revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in 
        for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground. 
        Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic 
        shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
        """

    if args.negative_prompt is None:
        args.negative_prompt = """
        Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, 
        low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, 
        misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
        """

    log_rank0(
        "=" * 20
        + " Prompts "
        + "=" * 20
        + f"\nPrompt: {args.prompt}\n\n"
        + f"Negative Prompt: {args.negative_prompt}"
    )

    # warm up
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=5.0,
        num_inference_steps=3,
    ).frames[0]

    torch.cuda.synchronize()
    s_t = time.time()

    reset_cache(pipe)
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=5.0,
        num_inference_steps=args.num_inference_steps,
    ).frames[0]

    torch.cuda.synchronize()
    e_t = time.time()
    log_rank0(f"Inference finished. Per prompt time:{(e_t - s_t):.2f}s")

    # Create parent directory for output file if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    export_to_video(output, args.output_file, fps=16)

    destroy_process_group()


if __name__ == "__main__":
    main()
