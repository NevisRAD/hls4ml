#!/usr/bin/env python3
"""Plot input / autoencoder-output pairs for cosmic, BNB, and nue examples.

Generates figure panels for fig:nueCCEVD in
  /workspace/E2E4RAD/sections/methods.tex

Three output PDFs are produced (one per event category):
  nueCCEVD_ae_pairs_cosmic.pdf
  nueCCEVD_ae_pairs_bnb.pdf
  nueCCEVD_ae_pairs_nue.pdf

Each PDF contains N_EXAMPLES rows with three columns:
  Input patch | AE reconstruction | Residual (input − output)

Usage (on the GPU machine):
  python plot_nueCCEVD_ae_pairs.py /path/to/autoencoder_model \
      [--output-dir /workspace/E2E4RAD/documents] \
      [--data-dir /nashome/s/sc5303/dune_data/RAD/ubOpenV2/data/ubOpen] \
      [--n-examples 4]

Based on workspace/uBOpenV2/notebook/plot_patches.py
Data: /nashome/s/sc5303/dune_data/RAD/ubOpenV2/data/ubOpen/
  neutrino_plane2_bg.h5        – cosmic (background)
  neutrino_plane2_sig_bnb.h5   – BNB signal
  neutrino_plane2_sig_nue.h5   – nue signal
"""

import argparse
import os

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import numpy as np
from tensorflow import keras

# ── default configuration ──────────────────────────────────────────────────────
_DEFAULT_DATA_DIR = '/nashome/s/sc5303/dune_data/RAD/ubOpenV2/data/ubOpen'
_DEFAULT_OUT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_N_EXAMPLES = 4

DATA_FILES = {
    'cosmic': 'neutrino_plane2_bg.h5',
    'bnb': 'neutrino_plane2_sig_bnb.h5',
    'nue': 'neutrino_plane2_sig_nue.h5',
}

LABEL_NAMES = {
    'cosmic': 'Cosmic (background)',
    'bnb': 'BNB signal',
    'nue': r'$\nu_e$ CC signal',
}

# HDF5 dataset key that stores the patch images
PATCH_KEY = 'data'


# ── helpers ────────────────────────────────────────────────────────────────────


def load_patches(h5_path, n, key=PATCH_KEY, seed=42):
    """Return *n* randomly-selected patches from *h5_path*.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    n : int
        Number of patches to sample.
    key : str
        Name of the dataset inside the HDF5 file.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n, H, W) or (n, H, W, 1)
    """
    with h5py.File(h5_path, 'r') as f:
        dset = f[key]
        n_tot = dset.shape[0]
        if n > n_tot:
            raise ValueError(
                f'Requested {n} examples but {h5_path} only contains {n_tot}.'
            )
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_tot, size=n, replace=False))
        return dset[idx]


def preprocess(patches):
    """Normalise each patch to [0, 1] using the 99th-percentile ADC value.

    Parameters
    ----------
    patches : np.ndarray, shape (N, H, W) or (N, H, W, 1)

    Returns
    -------
    np.ndarray, shape (N, H, W, 1), dtype float32
    """
    patches = patches.astype(np.float32)
    if patches.ndim == 3:
        patches = patches[..., np.newaxis]
    vmax = np.percentile(np.abs(patches), 99, axis=(1, 2, 3), keepdims=True)
    vmax = np.where(vmax == 0, 1.0, vmax)
    return patches / vmax


def reconstruct(model, patches):
    """Run *patches* through *model* and return reconstructions.

    Parameters
    ----------
    model : keras.Model
        Trained autoencoder.
    patches : np.ndarray, shape (N, H, W, 1)

    Returns
    -------
    np.ndarray, same shape as *patches*
    """
    return model.predict(patches, batch_size=len(patches), verbose=0)


def plot_input_output_pairs(inputs, outputs, label, save_path, cmap='viridis'):
    """Save a figure with *len(inputs)* rows of (input | AE output | residual).

    Parameters
    ----------
    inputs : np.ndarray, shape (N, H, W, 1)
    outputs : np.ndarray, shape (N, H, W, 1)
    label : str
        Category label used as figure title.
    save_path : str
        Destination PDF path.
    cmap : str
        Matplotlib colour map for the input and output panels.
    """
    n = inputs.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    fig.suptitle(f'{label}  --  input vs AE reconstruction', fontsize=13, y=1.01)

    col_titles = ['Input', 'AE output', 'Residual (input − output)']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11)

    for row in range(n):
        inp = inputs[row, ..., 0]
        out = outputs[row, ..., 0]
        residual = inp - out

        vmin_io = min(inp.min(), out.min())
        vmax_io = max(inp.max(), out.max())
        vext = max(abs(residual.min()), abs(residual.max()))
        # guard against a zero-extent colour range
        if vext == 0:
            vext = 1.0

        im0 = axes[row, 0].imshow(
            inp, aspect='auto', cmap=cmap, vmin=vmin_io, vmax=vmax_io, origin='lower'
        )
        im1 = axes[row, 1].imshow(
            out, aspect='auto', cmap=cmap, vmin=vmin_io, vmax=vmax_io, origin='lower'
        )
        im2 = axes[row, 2].imshow(
            residual, aspect='auto', cmap='RdBu_r', vmin=-vext, vmax=vext, origin='lower'
        )

        for ax in axes[row]:
            ax.set_xlabel('wire #', fontsize=8)
            ax.set_ylabel('time tick', fontsize=8)
            ax.tick_params(labelsize=7)

        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {save_path}')


# ── main ───────────────────────────────────────────────────────────────────────


def main(model_path, data_dir, output_dir, n_examples):
    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading model from {model_path} …')
    model = keras.models.load_model(model_path)
    model.summary()

    for tag, fname in DATA_FILES.items():
        h5_path = os.path.join(data_dir, fname)
        print(f'\n[{tag}]  loading {h5_path}')
        patches = load_patches(h5_path, n=n_examples)
        patches = preprocess(patches)
        recon = reconstruct(model, patches)

        save_path = os.path.join(output_dir, f'nueCCEVD_ae_pairs_{tag}.pdf')
        plot_input_output_pairs(
            patches,
            recon,
            label=LABEL_NAMES[tag],
            save_path=save_path,
        )

    print('\nDone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Plot input / AE-output pairs for cosmic, BNB, and nue patches. '
            'Generates one PDF per event category.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'model_path',
        help='Path to the trained Keras autoencoder (.h5 file or SavedModel dir).',
    )
    parser.add_argument(
        '--output-dir',
        default=_DEFAULT_OUT_DIR,
        help='Directory where the output PDF files are written.',
    )
    parser.add_argument(
        '--data-dir',
        default=_DEFAULT_DATA_DIR,
        help='Directory containing the HDF5 patch files.',
    )
    parser.add_argument(
        '--n-examples',
        type=int,
        default=_DEFAULT_N_EXAMPLES,
        help='Number of patch examples shown per event category.',
    )
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_examples=args.n_examples,
    )
