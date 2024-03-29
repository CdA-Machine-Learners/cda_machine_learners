

# Set up an Ubuntu CUDA server for LLM development
 - Expected hardware is a basic PC with an nvidia video card (recommended economical baseline is RTX 3090 ~$800 used at time of writing)
  ## Install Ubuntu Desktop 
  > - If adding Linux to an existing Windows server, create a new partition or separate drive for the Ubuntu Operating System
  > - Download latest Long Term Stable (LTS) release (ex. 22.04.4) installer ISO from: https://ubuntu.com/download/desktop
  > - Create a bootable USB drive from the ISO using Balena Etcher: https://etcher.balena.io/ 
  > - Optionally configure GRUB for boot management to select between Windows and Linux


  ## Install current NVIDIA native device drivers for target video card
  - https://www.nvidia.com/download/index.aspx?lang=en-us
   
  ## Build out CUDA environment
  
   ### Install the CUDA Toolkit from NVIDIA
    - https://developer.nvidia.com/cuda-toolkit-archive
   
   > Follow product selection flow for the desired release of the CUDA Toolkit (12.3.2)

   ### Install cuDNN
    - https://docs.nvidia.com/deeplearning/cudnn/installation/linux.html
