#ifndef DEEPFRAME_COMMON_H_
#define DEEPFRAME_COMMON_H_

#include <string>
#include <vector>



namespace deepframe {
  using std::string;
  // using std::

#define NOT_IMPLEMENT LOG(FATAL) << "Not Implemented.";

#define DISABLE_COPY_AND_ASSIGN(classname) \
  private :                                \
  classname(const classname&) ;            \
  classname& operator = (const classname&) \


}
#endif
