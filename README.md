
# Single-View 3D Hair Modeling with Clumping Optimization

Official code for the paper **"Single-View 3D Hair Modeling with Clumping Optimization"**.
For more details, visit the official paper website: [Clumping Hair](https://gapszju.github.io/ClumpingHair/)

> **Environment Information:**  
> This project has been tested on Windows with Visual Studio 2019.

## Environment Setup

1. **Install Pixi:**
   Please refer to the [Pixi official installation guide](https://pixi.sh/dev/installation/) to install Pixi, the environment manager

2. **Install Project Dependencies:**
   After installing Pixi, install the project dependencies.

   ```bash
   pixi install
   ```

3. **Run Post-installation Script:**
   Finalize the setup by running the post-installation script.

   ```bash
   pixi run postinstall
   ```

## Hair Reconstruction

1. **Calculate Initial Hair Strands:**
    Generate the initial hair strands from your data.

    ```bash
    pixi run gen_init_strands --root_real_imgs=<path-to-your-data>
    ```

2. **Optimize Hair Strands:**
    Optimize the generated hair strands.

    ```bash
    pixi run optimization --data=<path-to-your-data> --output=<path-to-your-output>
    ```

   - Replace `<path-to-your-data>` with the path to your input data.
   - Replace `<path-to-your-output>` with the directory where the optimized results will be saved.

