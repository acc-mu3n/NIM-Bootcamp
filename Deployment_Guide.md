# NIM Bootcamp

This bootcamp is designed to help developers get started with NVIDIA NIM™ by building real-world GenAI applications. The labs guide participants through setting up NIM Docker containers and utilizing REST API endpoints for serving inference requests. Additionally, attendees will explore fine-tuning models using Parameter Efficient Fine-Tuning (PEFT) techniques such as LoRA, with hands-on experience in fine-tuning adapters for the LLaMA-3 8B model.

## Deploying the labs

### Prerequisites

To run this tutorial, you will need a Laptop/Workstation/DGX machine with a minimum of 1(one) 80GB GPU of Ampere or later generation.

- Install the latest [Docker](https://docs.docker.com/engine/install/) and ensure that the installation includes NVIDIA Container Toolkit to enable GPU access.
- The finetuning lab requires a Huggingface security token to download model weights. Steps can be found [in the link here]( https://huggingface.co/docs/hub/en/security-tokens).


### Tested environment

We tested and ran all labs on a DGX machine equipped with an Ampere A100 GPU.


#### 1. Setting up a Virtual Environment

First, clone this repository and navigate to the project directory:
```bash
git clone https://github.com/openhackathons-org/NIM-Bootcamp/tree/main
cd NIM-Bootcamp
```

Create and activate a new virtual environment:
```bash
# Create virtual environment
python -m venv nim-bootcamp-env

# Activate virtual environment
source nim-bootcamp-env/bin/activate
```

#### 2. Installing Required Packages

With the virtual environment activated, install the required packages:
```bash
# Upgrade pip (recommended)
pip install --upgrade pip

# Install requirements
pip install -r https://github.com/openhackathons-org/NIM-Bootcamp/blob/main/bootcamp_requirements.txt  
```

#### 3. Verifying GPU Access

To verify that your environment can access GPU resources, run the following Python commands:
```python
import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# If CUDA is available, show device information
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name(0)}")
    print(f"Device count: {torch.cuda.device_count()}")
```

You can run these commands either in a Python terminal or by creating a simple script.

#### 4. Starting JupyterLab

To start JupyterLab on port 8888:
```bash
#Choose the desired workspace:
cd workspace-nim-with-rag
# Basic start
jupyter lab --port 8888

# If you want to make it accessible from other machines on your network
jupyter lab --port 8888 --ip 0.0.0.0

# If you want to specify a particular browser
jupyter lab --port 8888 --browser="chrome"
```

After running the command, you should see output similar to:
```
[I 2025-01-29 10:00:00.000 LabApp] JupyterLab extension loaded from /path/to/extension
[I 2025-01-29 10:00:00.000 LabApp] JupyterLab application directory is /path/to/app
[I 2025-01-29 10:00:00.000 ServerApp] Serving notebooks from local directory: /path/to/your/project
[I 2025-01-29 10:00:00.000 ServerApp] Jupyter Server 1.x is running at:
[I 2025-01-29 10:00:00.000 ServerApp] http://localhost:8888/lab
```

Copy the URL from the output and paste it into your browser. If prompted for a token, you can find it in the terminal output.

#### Troubleshooting

If you encounter any issues:

1. **Virtual Environment Issues**
   - Make sure you're in the correct directory when creating the virtual environment
   - Verify that the virtual environment is activated (you should see `(nim-bootcamp-env)` in your terminal prompt)

2. **Package Installation Issues**
   - Try updating pip before installing requirements: `pip install --upgrade pip`
   - If a package fails to install, try installing it separately

3. **GPU Access Issues**
   - Ensure NVIDIA drivers are properly installed
   - Check if CUDA toolkit is installed and matches your PyTorch version
   - Run `nvidia-smi` in terminal to verify GPU is recognized

4. **JupyterLab Access Issues**
   - Make sure port 8888 is not being used by another application
   - If accessing from another machine, ensure firewall settings allow the connection
   - Try a different port if 8888 is unavailable

For additional help, please open an issue in the GitHub repository.

Open the browser at `http://localhost:8888` and go click on the `Start_Here.ipynb`. As soon as you are done with the rest of the labs, shut down jupyter lab by selecting `File > Shut Down` and the container by typing `exit` or pressing `ctrl+d` in the terminal window.

Congratulations, you've successfully built and deployed an NIM Bootcamp!









