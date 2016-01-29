#include <iostream>
#include <string>

#include "json/json.h"

int main()
{
	
	//const char* str = "{\"succ\":true,\"code\":0,\"data\":\"http:\/\/inimg02.jiuyan.info\/in\/2015\/01\/10\/4DACCDE3-6727-0A43-6845-1002DAC81AC8.jpg\",\"timestamp\":\"1421306824\",\"_force\":false}";
	const char* str = "{\"succ\":true,\"code\":0,\"data\":{\"msg\":\"\"},\"timestamp\":\"1421307731\",\"_force\":false}";
	
  Json::Reader reader;
  Json::Value root;
  if (reader.parse(str, root)) 
  {
	  if(root["data"].isString()){
		  std::string sdata = root["data"].asString();
		  std::cout<<sdata<<std::endl;
	  }
	  else
		  std::cout<<"msg!"<<std::endl;
  }
  return 0;

}




