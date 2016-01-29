
#include <iostream>
#include <string>
#include <cstdio>

#include "downloader.h"

int main(int argc, char* argv[])
{
	if(argc < 2){
		std::cout<<"param error: "<<argv[0]<<" <url> "<<std::endl;
		return 0;
	}

	Downloader dd;
	//dd.set_proxy_host("192.168.2.150");
	//dd.set_proxy_port(2516);
	//dd.set_proxy("192.168.2.150:2516");
	std::string url(argv[1]);
	std::cout<<url<<std::endl;

    	
	//std::cout<<"get content: "<<std::endl;
	//std::string str = dd.get_content(url);
	//std::cout<<str<<std::endl;
	
	// http://inimg02.jiuyan.info/in/2014/10/25/A23A61E6-B87C-50DF-63A7-172D596E34F6.jpg
	
	std::cout<<"download file..."<<std::endl;
	dd.download_image_file(url, "output.jpg");
	std::cout<<"its done."<<std::endl;

	std::cout<<"wget file..."<<std::endl;
	std::string cmd = "wget " + url + " -O wget-output.jpg -q -o /dev/null ";
	FILE* fptr = popen(cmd.c_str(), "r");
	pclose(fptr);
	std::cout<<"its done."<<std::endl;
	
	return 0;
}
