#include "graph.h"

#ifdef WIN_COMPILE
#pragma warning(disable: 4661)
#endif

#ifdef WIN_COMPILE
#define LIBRARY_API __declspec(dllexport)
#else
#define LIBRARY_API
#endif

//#define LIBRARY_API __declspec(dllexport)
//#define LIBRARY_API

// Instantiations: <captype, tcaptype, flowtype>
// IMPORTANT: 
//    flowtype should be 'larger' than tcaptype 
//    tcaptype should be 'larger' than captype

template class LIBRARY_API Graph<int,int,int>;
template class LIBRARY_API Graph<short,int,int>;
template class LIBRARY_API Graph<float,float,float>;
template class LIBRARY_API Graph<double,double,double>;

