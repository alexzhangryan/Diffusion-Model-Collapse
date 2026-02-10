#!/opt/miniconda3/envs/edm/bin/python
"""Generate multiple images using the EDM model and calculate FID."""

import sys
import os
import pickle
import numpy as np
import scipy.linalg
import torch
import PIL.Image
import tqdm
from datetime import datetime

# Add the edm directory to the Python path
edm_path = os.path.join(os.path.dirname(__file__), 'edm')
sys.path.insert(0, edm_path)

import dnnlib

# ============================================================================
# HYPERPARAMETERS - Edit these
# ============================================================================
NUM_IMAGES = 10  # Number of images to generate
SEED_START = 0   # Starting seed
NUM_STEPS = 40   # Sampling steps (40 is recommended for AFHQ-v2)
BATCH_SIZE = 4   # Images per batch (lower for CPU)

# Paths
MODEL_PATH = os.path.expanduser('~/Downloads/edm-afhqv2-64x64-uncond-vp.pkl')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
FID_REF_URL = 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz'

# ============================================================================

def edm_sampler(
    net, latents, class_labels=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    """EDM sampler from generate.py"""
    # Adjust noise levels based on what's supported by the network
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # Main sampling loop
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # Increase noise temporarily
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def generate_images(network_pkl, outdir, seeds, batch_size, num_steps, device):
    """Generate images based on generate.py"""
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    # Generate images in batches
    print(f'Generating {len(seeds)} images...')
    all_seeds = list(seeds)
    for i in tqdm.tqdm(range(0, len(all_seeds), batch_size), desc='Batches'):
        batch_seeds = all_seeds[i:i + batch_size]

        # Set seeds and generate latents
        latents_list = []
        for seed in batch_seeds:
            torch.manual_seed(seed)
            latents = torch.randn([1, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            latents_list.append(latents)
        latents = torch.cat(latents_list, dim=0)

        class_labels = None

        # Generate images
        images = edm_sampler(net, latents, class_labels, num_steps=num_steps)

        # Save images
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_path = os.path.join(outdir, f'{seed:06d}.png')
            PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    print(f'Images saved to "{outdir}"')
    return net


def calculate_inception_stats(image_path, net_inception, device, num_images):
    """Calculate Inception statistics for generated images"""
    print(f'Loading images from "{image_path}"...')

    # Load images
    image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.png')])[:num_images]
    if len(image_files) < 2:
        raise ValueError(f'Found {len(image_files)} images, but need at least 2 to compute statistics')

    print(f'Calculating statistics for {len(image_files)} images...')

    feature_dim = 2048
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    for img_file in tqdm.tqdm(image_files, desc='Processing images'):
        # Load and preprocess image
        img = PIL.Image.open(os.path.join(image_path, img_file))
        img = np.array(img)

        # Convert to tensor and normalize to [0, 255]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).to(torch.uint8)

        # Get features
        features = net_inception(img_tensor, return_features=True).to(torch.float64)
        mu += features.squeeze()
        sigma += features.T @ features

    mu /= len(image_files)
    sigma -= mu.ger(mu) * len(image_files)
    sigma /= len(image_files) - 1

    return mu.cpu().numpy(), sigma.cpu().numpy()


def calculate_fid(mu, sigma, mu_ref, sigma_ref):
    """Calculate FID from inception statistics"""
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Check command-line arguments
    skip_generation = len(sys.argv) > 1 and sys.argv[1].lower() == 'no'

    if skip_generation:
        print('\n' + '='*60)
        print('SKIPPING GENERATION - Using existing images from output/')
        print('='*60 + '\n')

        # Count existing images
        existing_images = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
        num_existing = len(existing_images)
        print(f'Found {num_existing} existing images in {OUTPUT_DIR}')

        if num_existing < 2:
            print(f'\nError: Need at least 2 images for FID calculation, found {num_existing}')
            print('Run without arguments to generate new images first.')
            return
    else:
        # Generate images
        print('\n' + '='*60)
        print(f'GENERATING {NUM_IMAGES} NEW IMAGES')
        print('='*60 + '\n')

        seeds = range(SEED_START, SEED_START + NUM_IMAGES)
        generate_images(
            network_pkl=MODEL_PATH,
            outdir=OUTPUT_DIR,
            seeds=seeds,
            batch_size=BATCH_SIZE,
            num_steps=NUM_STEPS,
            device=device
        )
        num_existing = NUM_IMAGES

    print('\n' + '='*60)
    print('CALCULATING FID')
    print('='*60 + '\n')

    # Load Inception-v3 model for FID calculation
    print('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    with dnnlib.util.open_url(detector_url) as f:
        net_inception = pickle.load(f).to(device)

    # Calculate statistics for generated images
    mu_gen, sigma_gen = calculate_inception_stats(OUTPUT_DIR, net_inception, device, num_existing)

    # Load reference statistics
    print(f'Loading reference statistics from "{FID_REF_URL}"...')
    with dnnlib.util.open_url(FID_REF_URL) as f:
        ref = dict(np.load(f))

    # Calculate FID
    print('Calculating FID...')
    fid_score = calculate_fid(mu_gen, sigma_gen, ref['mu'], ref['sigma'])

    # Calculate additional statistics
    mu_gen_mean = np.mean(mu_gen)
    mu_gen_std = np.std(mu_gen)
    sigma_gen_trace = np.trace(sigma_gen)
    sigma_gen_det = np.linalg.det(sigma_gen)
    sigma_gen_frobenius = np.linalg.norm(sigma_gen, 'fro')

    # Reference statistics
    mu_ref_mean = np.mean(ref['mu'])
    mu_ref_std = np.std(ref['mu'])

    # Feature distribution comparison
    feature_distance = np.linalg.norm(mu_gen - ref['mu'])

    print('\n' + '='*60)
    print(f'FID Score: {fid_score:.2f}')
    print(f'(Based on {num_existing} images)')
    print('='*60)
    print('\nNote: FID requires ~50,000 images for accurate results.')
    print(f'With only {num_existing} images, this score has high variance and is not reliable.')

    # Save results to log file
    log_path = os.path.join(OUTPUT_DIR, 'log.txt')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(log_path, 'a') as f:
        f.write('\n' + '='*70 + '\n')
        f.write(f'Timestamp: {timestamp}\n')
        f.write('='*70 + '\n\n')

        f.write('MODEL PERFORMANCE METRICS\n')
        f.write('-'*70 + '\n')
        f.write(f'FID Score:                    {fid_score:.4f}\n')
        f.write(f'Number of Images:             {num_existing}\n')
        f.write(f'Sampling Steps:               {NUM_STEPS}\n')
        f.write(f'Feature Distance (L2):        {feature_distance:.4f}\n')
        f.write('\n')

        f.write('GENERATED FEATURES STATISTICS\n')
        f.write('-'*70 + '\n')
        f.write(f'Mean of Feature Means:        {mu_gen_mean:.4f}\n')
        f.write(f'Std of Feature Means:         {mu_gen_std:.4f}\n')
        f.write(f'Covariance Trace:             {sigma_gen_trace:.4f}\n')
        f.write(f'Covariance Determinant:       {sigma_gen_det:.4e}\n')
        f.write(f'Covariance Frobenius Norm:    {sigma_gen_frobenius:.4f}\n')
        f.write('\n')

        f.write('REFERENCE (REAL) FEATURES STATISTICS\n')
        f.write('-'*70 + '\n')
        f.write(f'Mean of Feature Means:        {mu_ref_mean:.4f}\n')
        f.write(f'Std of Feature Means:         {mu_ref_std:.4f}\n')
        f.write('\n')

        f.write('INTERPRETATION\n')
        f.write('-'*70 + '\n')
        if num_existing < 1000:
            f.write(f'⚠ WARNING: Only {num_existing} images used. Results are highly variable.\n')
            f.write('  Recommended: Use at least 1,000 images (preferably 50,000).\n')
        elif num_existing < 10000:
            f.write(f'⚠ CAUTION: {num_existing} images used. Results have moderate variance.\n')
            f.write('  Recommended: Use at least 10,000 images for stable estimates.\n')
        else:
            f.write(f'✓ {num_existing} images used. Results are reasonably stable.\n')

        f.write(f'\nLower FID is better (indicates generated images closer to real distribution)\n')
        f.write(f'Typical FID for AFHQ-v2 64x64 with EDM: ~2.0 (with 50k images)\n')
        f.write('\n')

    print(f'\n✓ Statistics saved to: {log_path}')

    # Usage instructions
    print('\n' + '='*60)
    print('USAGE:')
    print('  python generate_image.py       # Generate new images + calculate FID')
    print('  python generate_image.py no    # Skip generation, use existing images')
    print('='*60)


if __name__ == "__main__":
    main()
