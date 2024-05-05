import os
from tqdm import tqdm
from pipe import DDPMPipeline
from sch import DDPMScheduler as customized_scheduler

mid = 'google/ddpm-cifar10-32'
# mid = 'anton-l/ddpm-butterflies-128'
gen1 = DDPMPipeline.from_pretrained(mid).to("mps")
gen1.scheduler = customized_scheduler(num_train_timesteps=1000, extra_gradient=False)
# print(gen1.scheduler)

#steps = [2, 5, 10, 25, 50, 100, 200, 500]
steps = [5]

for si in steps:
    for i in tqdm(range(0, 1), desc='Generating'):
        if not os.path.exists(f'cifar_ddpm/{si}'):
            os.makedirs(f'cifar_ddpm/{si}')
        img1 = gen1(num_inference_steps=si).images
        img1[0].save(f'cifar_ddpm/{si}/bfly_{i:04}.png')