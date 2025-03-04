# SimpleClassifier
This simple classifier creates a API for interacting with YOLO models. 

## Prerequisites

Before building the project, ensure you have the following installed on your system:

- **Arch Linux** (or any Linux distribution)

### Install Dependencies

On Arch Linux, you can install the required dependencies using `pacman`:

```bash
sudo pacman -Syu
sudo pacman -S base-devel cmake glfw-x11 glm
```

For Wayland support, install `glfw-wayland` instead of `glfw-x11`.

---

## Project Structure

The project has the following structure:

```
glfw-template/
├── include/
│   └── glad/
│       └── glad.h          # GLAD header
├── src/
|   ├── main.cc                # Main application code
│   └── glad.c              # GLAD source file
├── CMakeLists.txt          # CMake build configuration
└── README.md               # This file
```

---

## Setting Up GLAD

GLAD is used to load OpenGL functions. The necessary files (`glad.h` and `glad.c`) are already included in this project. If you need to regenerate them:

1. Visit the [GLAD Web Service](https://glad.dav1d.de/).
2. Configure the settings:
   - **Language**: C++
   - **Specification**: OpenGL
   - **API**: gl: Version 4.6 (or your desired version)
   - **Profile**: Core
   - **Generate a loader**: Check this option
3. Download the generated ZIP file and replace the `include/glad` and `src/glad.c` files in the project.

---

## Building the Project

To build the project, follow these steps:

1. Clone the repository and navigate to the cloned directory:
  
   ```bash
   git clone https://github.com/zerootoad/glfw-template.git
   cd glfw-template
   ```

2. Create a `build` directory and navigate into it:

   ```bash
   mkdir build
   cd build
   ```

3. Run CMake to configure the project:

   ```bash
   cmake ..
   ```

   This will generate the necessary build files.

4. Build the project using `make`:

   ```bash
   make
   ```

5. Run the executable:

   ```bash
   ./LearningGLFW
   ```

---

## CMake Configuration

The `CMakeLists.txt` file is configured as follows:

- It sets the C++ standard to **C++17**.
- It includes the `include` directory for GLAD and GLM.
- It links the **GLFW** and **OpenGL** libraries.

If you need to modify the CMake configuration, edit the `CMakeLists.txt` file.

---

## Troubleshooting

### 1. **Header File Not Found**
If your editor (e.g., Helix, VSCode) cannot find the `glad/glad.h` header file, ensure that the `compile_commands.json` file is generated if not, double-check for the following line in your `CMakeLists.txt`:

```cmake
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
```

Then, reconfigure and build the project:

```bash
cd build
cmake ..
make
```

This will generate a `compile_commands.json` file in the `build` directory, which your editor can use to resolve include paths.

### 2. **GLFW Not Found**
Ensure that GLFW is installed correctly. If you encounter issues, try reinstalling it:

```bash
sudo pacman -S glfw-x11 
```
For Wayland support, reinstall `glfw-wayland` instead of `glfw-x11`.

---

## Acknowledgments

- [GLAD](https://glad.dav1d.de/) for OpenGL function loading.
- [GLFW](https://www.glfw.org/) for window and input management.
- [GLM](https://github.com/g-truc/glm) for OpenGL mathematics.

---

Enjoy building your OpenGL applications! If you have any questions or issues, feel free to open an issue.
