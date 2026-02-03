"""Wan Studio entrypoint.

We install a few safe defaults before importing Qt / diffusers:
- Silence a known-noisy Wan LoRA warning from diffusers.
- Set PYTORCH_ALLOC_CONF default to reduce CUDA fragmentation.
"""

from wan_studio.logging_setup import install_env_defaults, install_log_filters

install_env_defaults()
install_log_filters()

from wan_studio.gui import main

if __name__=='__main__':
    main()
