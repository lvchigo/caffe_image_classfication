/*
 * =====================================================================================
 *
 *       filename:  API_data_augmentation.h
 *
 *    description:  caffe for data augmentation interface
 *
 *        version:  1.0
 *        created:  2016-07-20
 *       revision:  none
 *       compiler:  g++
 *
 *         author:  xiaogao
 *        company:  in66.com
 *
 *      copyright:  2016 itugo Inc. All Rights Reserved.
 *      
 * =====================================================================================
 */


#ifndef _API_DATA_AUGMENTATION_H_
#define _API_DATA_AUGMENTATION_H_

#include <opencv/cv.h>
#include <vector>

using namespace cv;
using namespace std;


class API_DATA_AUGMENTATION
{

/***********************************public***********************************/
public:

    /// construct function 
    API_DATA_AUGMENTATION();
    
	/// distruct function
	~API_DATA_AUGMENTATION(void);

	/***********************************Get_Neg_Patch**********************************/
	int Get_Neg_Patch(
		vector < pair < string,Vec4i > > 	vecInRect,			//[In]:Input Rect
		string								patchName,			//[In]:Patch Name
		string								negName,			//[In]:Get Neg Patch Name
		const int 							width,				//[In]:image.width
		const int							height,				//[In]:image.height
		const float							T_IoU,				//[In]:T_IoU
		vector < pair < string,Vec4i > > 	&vecOutRect);		//[Out]:Res

/***********************************private***********************************/
private:	

	/***********************************Rect_IOU**********************************/
	int Rect_IOU(Vec4i rect1, Vec4i rect2, float &iou);

};

#endif

