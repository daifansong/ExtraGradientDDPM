from pipeline import DDPMPipeline
from scheduler import DDPMScheduler as customized_scheduler
import argparse

parser = argparse.ArgumentParser(description="Original and Extra Gradient DDPM")
parser.add_argument('--extra_gradient', type=bool, default=False, help='Extra Gradient or Standard DDPM')
parser.add_argument('--inference_steps', type=int, default=50, help='Inference Steps')

args = parser.parse_args()

# mid = 'google/ddpm-cifar10-32'
mid = 'anton-l/ddpm-butterflies-128'
gen1 = DDPMPipeline.from_pretrained(mid).to("mps")
gen1.scheduler = customized_scheduler(num_train_timesteps=1000, extra_gradient=args.extra_gradient)

img1 = gen1(num_inference_steps=args.inference_steps).images
img1[0].save(f'Infer{args.inference_steps}.png')
