#pragma once

#include <GL/glew.h>
#include <stdio.h>
#include <stdlib.h>


inline void getLastGlError(const char *errorMessage, const char *file, const int line) 
{                               
  GLenum gl_error = glGetError();

  if (gl_error != GL_NO_ERROR) 
  {
      fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
      fprintf(stderr, "%s (%s)", gluErrorString(gl_error), errorMessage);
      fprintf(stderr, ", id: %d\n", gl_error);
      exit(EXIT_FAILURE);
  }
}

#define GET_GL_ERROR(msg) getLastGlError(msg, __FILE__, __LINE__);