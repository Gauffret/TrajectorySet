/*
* @INPROCEEDINGS{Wang2013,
* 	author={Heng Wang and Cordelia Schmid},
* 	title={Action Recognition with Improved Trajectories},
* 	booktitle= {IEEE International Conference on Computer Vision},
* 	year={2013},
* 	address={Sydney, Australia},
* 	url={http://hal.inria.fr/hal-00873267}
* }
*/

#include "../include/DenseTrackStab.h"
#include "../include/Initialize.h"
#include "../include/Descriptors.h"
#include "../include/OpticalFlow.h"
#include <iostream>
#include <fstream>

#include <time.h>

using namespace cv;

#define CELL_SIZE   10       //セルサイズ (pixel)
#define BLOCK_SIZE  5       //ブロックサイズ (セル)
// #define GAP  10       	    //ギャップ

clock_t start = clock();    // スタート時間


int show_track = 0; // set show_track = 1, if you want to visualize the trajectories

int main(int argc, char** argv)
{
//   int C =0;
	std::ofstream fout;
	fout.open(argv[2], std::ios::out|std::ios::binary|std::ios::trunc);
    //  ファイルを開く
    //  ios::out は書き込み専用（省略可）
    //  ios::binary はバイナリ形式で出力（省略するとアスキー形式で出力）
    //  ios::truncはファイルを新規作成（省略可）
    //  ios::addにすると追記になる
      
    if (!fout) {
        std::cout << "ファイル file.txt が開けません"<<std::endl;
        return 1;
    }
        //  ファイルが開けなかったときのエラー表示
        
        
	VideoCapture capture;
	char* video = argv[1];
	int flag = arg_parse(argc, argv);
	capture.open(video);

	if(!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}

	int frame_num = 0;
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;

	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);
// std::cout<<"test5"<<std::endl;  
	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo, video);
// std::cout<<"test5"<<std::endl;  
	std::vector<Frame> bb_list;
	if(bb_file) {
		LoadBoundBox(bb_file, bb_list);
		assert(bb_list.size() == seqInfo.length);
	}

	if(flag)
		seqInfo.length = end_frame - start_frame + 1;

//	fprintf(stderr, "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);
 
	if(show_track == 1)
		namedWindow("DenseTrackStab", 0);

	SurfFeatureDetector detector_surf(200);
	SurfDescriptorExtractor extractor_surf(true, true);

	std::vector<Point2f> prev_pts_flow, pts_flow;
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<Point2f> prev_pts_all, pts_all;
	
	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;
	Mat flow, human_mask;

	Mat image, prev_grey, grey;

	std::vector<float> fscales(0);	//スケール[x]の時に何分の1倍にしているか
	std::vector<Size> sizes(0);	//スケール[x]の時のサイズ

	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0), flow_warp_pyr(0);
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0), poly_warp_pyr(0);

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points
	while(true) {

		Mat frame;
		int i, j, c;	
		
		// get a new frame
		capture >> frame;
				
		if(frame.empty())
			break;

		if(frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}

		if(frame_num == start_frame) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);

			InitPry(frame, fscales, sizes);

			BuildPry(sizes, CV_8UC1, prev_grey_pyr);
			BuildPry(sizes, CV_8UC1, grey_pyr);
			BuildPry(sizes, CV_32FC2, flow_pyr);
			BuildPry(sizes, CV_32FC2, flow_warp_pyr);

			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_warp_pyr);

			xyScaleTracks.resize(scale_num);
		
			frame.copyTo(image);
			cvtColor(image, prev_grey, CV_BGR2GRAY);

			//様々なスケールで特徴点を検出（スケールが違っても特徴点を被らないように）
			for(int iScale = 0; iScale < scale_num; iScale++) {
				if(iScale == 0)
					prev_grey.copyTo(prev_grey_pyr[0]);
				else
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

				//test//
// 				std::cout<<iScale<<std::endl;
				// dense sampling feature points
				std::vector<Point2f> points(0);
				//test
// 				std::cout<<"pointsize "<<points.size()<<std::endl;
				DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);

				// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];
				
				
				for(i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
// 					std::cout<<points[i]<<std::endl;
// 					std::cout<<"pointsize "<<points.size()<<std::endl;
					//test//
// 				printf("%f\t%f\t\n",points[i].x,points[i].y);
			}

			// compute polynomial expansion
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);

			human_mask = Mat::ones(frame.size(), CV_8UC1);
			if(bb_file)
				InitMaskWithBox(human_mask, bb_list[frame_num].BBs);

			detector_surf.detect(prev_grey, prev_kpts_surf, human_mask);
			extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);

			frame_num++;
			continue;
		}
		//test//
// 		std::cout<<frame_num<<std::endl;
		
		init_counter++;
		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

		// match surf features
		if(bb_file)
			InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
		detector_surf.detect(grey, kpts_surf, human_mask);
		extractor_surf.compute(grey, kpts_surf, desc_surf);
		ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

		MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow, human_mask);
		MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);

		Mat H = Mat::eye(3, 3, CV_64FC1);
		if(pts_all.size() > 50) {
			std::vector<unsigned char> match_mask;
			Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
			if(countNonZero(Mat(match_mask)) > 25)
				H = temp;
		}
		
		Mat H_inv = H.inv();
		Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);
		MyWarpPerspective(prev_grey, grey, grey_warp, H_inv); // warp the second frame

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey_warp, poly_warp_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_warp_pyr, flow_warp_pyr, 10, 2);
		
		int xyScaleTracksALL = 0;
		if(frame_num>14){
		  for(int i = 0;i<scale_num;i++)
		    xyScaleTracksALL += xyScaleTracks[i].size();
		}		
// 		std::cout<<"all "<<xyScaleTracksALL<<std::endl;
		std::vector<Point2f> trajtrack;
		std::vector<vector<Point2f> > traLen(xyScaleTracksALL);
		std::vector<vector<Point2f> > traPoint(xyScaleTracksALL);
		int PointNumber = 0;
		
		
		for(int iScale = 0; iScale < scale_num; iScale++) {
			if(iScale == 0)
				grey.copyTo(grey_pyr[0]);
			else
				resize(grey_pyr[iScale-1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

			int width = grey_pyr[iScale].cols;
			int height = grey_pyr[iScale].rows;

// 			// compute the integral histograms
// 			DescMat* hogMat = InitDescMat(height+1, width+1, hogInfo.nBins);
// 			HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo);
// 
// 			DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
// 			HofComp(flow_warp_pyr[iScale], hofMat->desc, hofInfo);
// 
// 			DescMat* mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);
// 			DescMat* mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);
// 			MbhComp(flow_warp_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);

			// track feature points in each scale separately
			std::list<Track>& tracks = xyScaleTracks[iScale];
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
				int index = iTrack->index;
				//test//
// 				std::cout<<"indexbe "<<index<<std::endl;
				Point2f prev_point = iTrack->point[index];
// 				std::cout<<"POINT "<<iTrack->point[1]<<std::endl;

				//test//
// 				printf("%f\t%f\t\n",iTrack->point[index].x,iTrack->point[index].y);
				
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];

// 				//test//
// 				std::cout<<"flow "<<flow_pyr[iScale].ptr<float>(y)[2*x]<<std::endl;
// 				
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);
					continue;
				}

				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];

				// get the descriptors for the feature point
				RectInfo rect;
// 				GetRect(prev_point, rect, width, height, hogInfo);
// 				GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
// 				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
// 				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
// 				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				iTrack->addPoint(point);
				//test//
// 				std::cout<<"indexaf "<<iTrack->index<<std::endl;
				
				// draw the trajectories at the first scale
				if(show_track == 1 && iScale == 0)
					DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);

				// if the trajectory achieves the maximal length
				if(iTrack->index >= trackInfo.length) {

				    //lengeth+1個目の座標がわかってないと，各ギャップlength個出せない
					std::vector<Point2f> trajectory(trackInfo.length+1);
					for(int i = 0; i <= trackInfo.length; ++i){
						trajectory[i] = iTrack->point[i]*fscales[iScale];
// 						if(i != trackInfo.length)
// 						traPoint.push_back(trajectory[i]);
					}
											//test//
// 						std::cout<<"framenumber "<<frame_num<<std::endl;
// 						std::cout<<"tra "<<traPoint.size()<<std::endl;

					
					std::vector<Point2f> displacement(trackInfo.length);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i]*fscales[iScale];		
										
					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
					if(IsValid(trajectory,trajtrack, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement)) {
						// output the trajectory
// 						printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t", frame_num,iTrack->index, mean_x, mean_y, var_x, var_y, length, fscales[iScale]);
						
						//test//
// 						std::cout<<"break"<<" "<<"trackInfo_length "<<trackInfo.length<<std::endl;
					
						

// 						// for spatio-temporal pyramid
// 						printf("%f\t", std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999));
// 						printf("%f\t", std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999));
// 						printf("%f\t", std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999));
					
						
						//test//
// 						std::cout<<"break_spatiotemporal_pyramid"<<std::endl;
// 						std::cout<<xyScaleTracks.size()<<frame_num<<std::endl;
						// output the trajectory////////////////////////
						for (int i = 0; i < trackInfo.length; ++i){
						  traLen[PointNumber].push_back(displacement[i]);
						  traPoint[PointNumber].push_back(trajtrack[i]);
// 							printf("%f\t%f\t\n",traPoint[i*frame_num].x,trajectory[i].y);
						}
						PointNumber ++;
// 						/////////////////////////////////////////////
						
// 						printf("%f\t%f\t%f\t%f\t",trajectory[i].x,trajectory[i].y, displacement[i].x, displacement[i].y);
// 						printf("%f\t%f\t",displacement[i].x, displacement[i].y);

// 						
// 						//test//
// // 						std::cout<<"break_output_trajectory"<<std::endl;
// 
// 						PrintDesc(iTrack->hog, hogInfo, trackInfo);	/*std::cout<<"break"<<std::endl;*/
// 						PrintDesc(iTrack->hof, hofInfo, trackInfo);	/*std::cout<<"break"<<std::endl;*/
// 						PrintDesc(iTrack->mbhX, mbhInfo, trackInfo);	/*std::cout<<"break"<<std::endl;*/
// 						PrintDesc(iTrack->mbhY, mbhInfo, trackInfo);	
// 						printf("\n");
					}

					iTrack = tracks.erase(iTrack);
					continue;
				}
				++iTrack;
			}
// 			ReleDescMat(hogMat);
// 			ReleDescMat(hofMat);
// 			ReleDescMat(mbhMatX);
// 			ReleDescMat(mbhMatY);

			if(init_counter != trackInfo.gap)
				continue;

			// detect new feature points every gap frames
			//オーバーラップしている点を検出して加えないようにする
			std::vector<Point2f> points(0);		
			for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
			  
				points.push_back(iTrack->point[iTrack->index]);
		
			DenseSample(grey_pyr[iScale], points, quality, min_distance);
			// save the new feature points
			for(i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		}

		init_counter = 0;
		grey.copyTo(prev_grey);
		for(i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}
		
// 		std::cout<<"test"<<std::endl; 
// ////////////////////領域を作成する///
// 	    std::cout<<"framenumber "<<frame_num<<std::endl;
// 	    std::cout<<"PointNumber "<<PointNumber<<std::endl;
	if(frame_num > 14 && PointNumber > 0){
	  

//  	  int width = 50;
// 	  int height = 50;

	  int width = grey_pyr[0].cols;
	  int height = grey_pyr[0].rows;

	  //すべての点がどのセルに入っているかを記録,点の含まれているセルの番号（座標）を記録したベクトル
	  std::vector<Point2f> all_point(PointNumber);
	  //画像サイズの2次元配列をCELLSIZEで分割したもの．2次元座標のセルに入っている点の番号を（複数個）記録するので3次元配列
	  std::vector<vector<vector<int> > > pointINcell(width, vector<vector<int> >(height));

	  //どの点がpoinINcell2次元配列のどこに入っているかを記録するためにCELLSIZEで割って切り捨てして整数化
	  for(int l = 0; l < PointNumber; l++){ 
	    all_point[l].x = std::min<int>(std::max<int>(cvFloor(traPoint[l][0].x/CELL_SIZE), 0), width-1);
	    all_point[l].y = std::min<int>(std::max<int>(cvFloor(traPoint[l][0].y/CELL_SIZE), 0), height-1);
	    pointINcell[all_point[l].x][all_point[l].y].push_back(l);
	  }

		    //ブロックの移動		    
		    for(int y = 0; y < ((height/CELL_SIZE) - BLOCK_SIZE + 1); y++){
			for(int x = 0; x < ((width/CELL_SIZE) - BLOCK_SIZE +1 ); x++){
			  int NoneCount = 0;//すべてのセルが0の時出力しないための変数
			  int cellnumber = 0;//何番目のセルか
			    //セル内の移動
			    for(j = 0; j < BLOCK_SIZE; j++){
				for(i = 0; i < BLOCK_SIZE; i++){
// 				  std::cout<<i<<j<<"pointINcell "<<pointINcell[x+i][y+j].size()<<std::endl;
				  std::vector<Point2f> cell (track_length);
				  std::vector<float> cellmean (track_length*25);

				  //セル内に複数個の点が入っていた時，その平均の軌跡を出力
				  if(pointINcell[x+i][y+j].size() > 0){
				      if(NoneCount > 0){
					  for(int m = 0; m < track_length * NoneCount; m++){
					    cellmean[m] = 0;
// 					    printf("%f\t%f\t",cellmean[m],cellmean[m]) ;
					    fout.write(( char * ) &cellmean[m],sizeof( float ) );
					    fout.write(( char * ) &cellmean[m],sizeof( float ) );
// 					    C++;C++;
					  }/*C += NoneCount;*/
					  NoneCount = 0;
				      }
				      for(int m = 0; m < track_length; m++){
					for(int l = 0; l < pointINcell[x+i][y+j].size();l++){
					    cell[m].x  += traLen [pointINcell[x+i][y+j][l]] [m].x ;/**250;*///char型に入れるために250倍
					    cell[m].y  += traLen [pointINcell[x+i][y+j][l]] [m].y ;/**250;*/
  // 					  std::cout<<"cell "<<cell[m]<<std::endl;
					}
				      } 
				      for(int m = 0; m < track_length; m++){
					cellmean[m*2] = cell[m].x / pointINcell[x+i][y+j].size();
					cellmean[m*2 +1] = cell[m].y / pointINcell[x+i][y+j].size();
//   				      printf("%f\t%f\t",cellmean[m*2], cellmean[m*2 +1]) ;
  				      fout.write(( char * ) &cellmean[m*2], sizeof( float ) );
  				      fout.write(( char * ) &cellmean[m*2+1], sizeof( float ) );

				      //文字列ではないデータをかきこむ
				      // 「sizeof( double )」バイトの「char *」ポインタ「a[i]」をデータとして出力
	/*C++;	C++;*/		      }/*C++;*/
				  }
				  //セル内に点が一つも入っていない時０を出力
				  //すべてのセルが0の時出力しない
				  else{
				      if(NoneCount == cellnumber){ 
					NoneCount++;	//何も入っていない時をカウント
				      }
				      else{
					for(int m = 0; m < track_length * 2; m++){
					  cellmean[m] = 0;
// 					  printf("%f\t",cellmean[m]) ;
					  fout.write(( char * ) &cellmean[m],sizeof( float ) );
	/*C++;*/			}/*C++;*/
				      }
				  }cellnumber++;
				}
			    }
// 			   //最後のセルの時
// 			    if(NoneCount != 25)	//すべてのセルが0の時出力しない			    
// 			      printf("\n");
			}
		    }
	}
// 	//////////////////////////	      
// 		if(frame_num > 240)
// 		std::cout<<"test2"<<std::endl;   
		
		
		prev_kpts_surf = kpts_surf;
		desc_surf.copyTo(prev_desc_surf);

		frame_num++;

// 		clock_t half = clock();     // 終了時間
// 		std::cout << "duration = " << (double)(half - start) / CLOCKS_PER_SEC << "sec.\n";

		
		if( show_track == 1 ) {
			imshow( "DenseTrackStab", image);
			c = cvWaitKey(3);
			if((char)c == 27) break;
		}
	}
// 	 std::cout << init_counter<<std::endl;


	  clock_t end = clock();     // 終了時間
	  std::cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";
// 	  std::cout<<"N"<<C<<std::endl;

	  
	  if( show_track == 1 )
		destroyWindow("DenseTrackStab");

	      fout.close();  //ファイルを閉じる
	return 0;
}
