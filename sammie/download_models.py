import os
import requests
import hashlib
from tqdm import tqdm


# Single-file checkpoints downloaded by direct URL
_links = [
    ('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt', '2b30654b6112c42a115563c638d238d9', 'checkpoints'),
    ('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt', 'ec7bd7d23d280d5e3cfa45984c02eda5', 'checkpoints'),
    ('https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_s_512x512.pt', '962e151a9dca3b75d8228a16e5264010', 'checkpoints'),
    ('https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth', 'a50eeaa149a37509feb45e3d6b06f41d', 'checkpoints'),
    ('https://huggingface.co/zibojia/minimax-remover/resolve/main/transformer/diffusion_pytorch_model.safetensors','183c7a631e831f73f8da64c5c4d83e2f', 'checkpoints/transformer'),
    ('https://huggingface.co/zibojia/minimax-remover/resolve/main/vae/diffusion_pytorch_model.safetensors','3f80444947443d8f36c0ed2497c20c8d', 'checkpoints/vae'),
]

# Multi-file HuggingFace model repos (diffusers pipelines with configs + weights)
# These cannot be reduced to single URLs — snapshot_download fetches the full repo structure
# (repo_id, local_dir)
_hf_repos = [
    ('SammyLim/VideoMaMa',                             'checkpoints/videomama_unet'),
    ('stabilityai/stable-video-diffusion-img2vid-xt',  'checkpoints/stable-video-diffusion-img2vid-xt'),
]


def download_models():
    # --- Single-file direct downloads (existing pattern) ---
    for link, md5, directory in _links:
        os.makedirs(directory, exist_ok=True)

        filename = link.split('/')[-1]
        filepath = os.path.join(directory, filename)

        if not os.path.exists(filepath) or hashlib.md5(open(filepath, 'rb').read()).hexdigest() != md5:
            print(f'Downloading {filename} to {directory}...')
            r = requests.get(link, stream=True)
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024
            t = tqdm(total=total_size, unit='iB', unit_scale=True)
            with open(filepath, 'wb') as f:
                for data in r.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()
            if total_size != 0 and t.n != total_size:
                raise RuntimeError('Error while downloading %s' % filename)
        else:
            print(f'{filename} already downloaded in {directory}.')

    # --- Multi-file HuggingFace repos (VideoMaMa UNet + SVD base model) ---
    from huggingface_hub import snapshot_download
    for repo_id, local_dir in _hf_repos:
        os.makedirs(local_dir, exist_ok=True)
        if not os.listdir(local_dir):
            print(f'Downloading HF repo {repo_id} to {local_dir}...')
            snapshot_download(repo_id=repo_id, local_dir=local_dir)
        else:
            print(f'{local_dir} already downloaded.')


if __name__ == '__main__':
    download_models()