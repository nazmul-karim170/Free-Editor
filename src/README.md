## Step#1 Create environment  

Do the Following-

    ```bash
        conda create --name nerfstudio -y python=3.9
        conda activate nerfstudio
        python -m pip install --upgrade pip
    ```

## Step#2 Install dependencies

    1. If you have exisiting installation, first make sure to uninstall using this command:

    ```bash
        pip uninstall torch torchvision functorch tinycudann
    ```

    2. Then Install CUDA 11.8 with this command:

    ```bash
        conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
    ```

    3. Then install Pytorch 2.1.2 using this command:

    ```bash
        pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    ```

# Step#3 Install tiny-cuda-nn/gsplat (Use bash file "install_packages.sh" to install tinycuda on server)

    After pytorch and ninja, install the torch bindings for tiny-cuda-nn:

    ```bash
        pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    ```

# Step#4 Installing nerfstudio

    1. Sometimes, you may face issue with configobj or other packages, manually install them from source. For example,

    ```bash
        git clone https://github.com/DiffSK/configobj.git
        cd configobj
        python setup.py install
    ```

    2. From pip (it does not work in cluster):

    ```bash
        pip install nerfstudio
    ```

    3. If you want build from source and want the latest development version, use this command:

    ```bash
        git clone https://github.com/nerfstudio-project/nerfstudio.git
        cd nerfstudio
        pip install --upgrade pip setuptools
        pip install -e .
    ```

# Step#5 Validate NerfStudio Installation


    1. Download some test data: (for cluster use-- curl -x http://StripDistrict:10132 -O https://path-to-nerfstudio-data/nerfstudio_poster_data.zip)

    ```bash
        ns-download-data nerfstudio --capture-name=poster           
    ```

    2. Train model

    ```bash
        ns-train nerfacto --data data/nerfstudio/poster
    ```

    If you start seeing on your linux terminal that it started training, then it means everything is good to go!


# Step#6 Install the Nerf Baselines for downloading datasets and valdiating other methods

    1. For this, you have to create a new environment for this as there are package conflicts with NerfStudio. We choose gaussian splatting here.

        ```bash
            git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
            mv gaussian-splatting/ gaussian_splatting
            cd gaussian_splatting


            conda env create --file environment.yml
            conda activate gaussian_splatting
            pip install simple-kNN
        ```

    2. Then, do pip installation.

        ```bash
            pip install nerfbaselines
        ```

    3. Download Some Datasets. For Example,

        i. Downloads the garden scene to the cache folder.

            ```bash
                mdkir data
                cd data
                mkdir nerf_dataset
                cd nerf_dataset
                
                nerfbaselines download-dataset external://mipnerf360/garden
            ```

        ii. Downloads all nerfstudio scenes to the cache

            ```bash
                nerfbaselines download-dataset external://nerfstudio
            ```

        iii. Downloads kithen scene to folder kitchen

            ```bash
                nerfbaselines download-dataset external://mipnerf360/kitchen -o kitchen
            ```

    4. To download other datasets, please visit this link - https://huggingface.co/datasets/yangtaointernship/RealEstate10K-subset/tree/main

        i. Here, "synthetic_scenes.zip" is the deepvoxels data.

        i. "nerf_synthetic" and blender dataset possibly the same dataset.
        
        ii. "frames.zip" is the extracted frames for 200 scenes of RealEstate10K dataset. "train.zip" is the camera files. 

        iii. Extra: if needs to be trasnferred to a cluster, Extract the "frames.zip" and tranfer using which will only files that are not already transferred

            ```bash
                rsync -avz --progress /path/to/local/folder/ username@remote_host:/path/to/remote/folder/
            ```
        iv. For Shiny Dataset, go to - https://nex-mpi.github.io/

        v. For Spaces Dataset, 

            ```bash
                git clone https://github.com/augmentedperception/spaces_dataset
            ```

## Download Pre-trained Models 

    * Caption Generation Model 

        ```bash
            git clone https://huggingface.co/Salesforce/blip2-opt-2.7b
        ```     

    * Stable Diffusion 3 Medium (Fast and Accurate)

        ```bash
            git clone https://huggingface.co/stabilityai/stable-diffusion-3-medium
        ```        

    * If you don't want to download the pre-trained model, generate an access token in hugging face (Go to your account settings) and login into your account 

        ```bash
            huggingface-cli login
        ``` 