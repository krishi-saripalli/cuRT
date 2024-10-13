#include "window.h"
#include <iostream>

Window::Window(int w, int h, std::string name) : width{w}, height{h}, windowName{name} {
    std::cout << "Window Constructed!" << std::endl;
    initWindow();
}

Window::~Window() {
    std::cout << "Window Destroyed!" << std::endl;
    glfwDestroyWindow(glWindow);
    glfwTerminate();
}
void Window::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    glWindow = glfwCreateWindow(width,height,windowName.c_str(),nullptr,nullptr);

    if (!glWindow) 
    {
        fprintf(stderr, "Failed to create a Window\n");
        glfwTerminate();
        exit(-1);
    }

    //make context current so we can bind textures to the window
    glfwMakeContextCurrent(glWindow);

    if (glewInit() != GLEW_OK) 
    {
        fprintf(stderr, "Failed to initialize GLEW\n");
        exit(-1);
    }
   
    glfwShowWindow(glWindow);
    
}