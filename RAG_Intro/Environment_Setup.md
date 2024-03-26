

# Set up an Ubuntu CUDA server for LLM development
  ## Install Ubuntu Desktop 
  > - If adding Linux to an existing Windows server, create a new partition or separate drive for the Ubuntu Operating System
  > - Download latest Long Term Stable (LTS) release (ex. 22.04.4) installer ISO from: https://ubuntu.com/download/desktop
> 
  > - Optionally configure GRUB for boot management to select between Windows and Linux


  ## Install current NVIDIA native device drivers for target video card
  - https://www.nvidia.com/download/index.aspx?lang=en-us
   
  ## Build out CUDA environment
  
   ### Install the CUDA Toolkit from NVIDIA
    - https://developer.nvidia.com/cuda-toolkit-archive
   
   > Follow product selection flow for the desired release of the CUDA Toolkit (12.3.2)

   ### Install cuDNN
    - https://docs.nvidia.com/deeplearning/cudnn/installation/linux.html
