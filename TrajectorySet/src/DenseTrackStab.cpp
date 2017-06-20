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

#define CELL_SIZE   10 
#define BLOCK_SIZE  5

clock_t start = clock();


int show_track = 0; // set show_track = 1, if you want to visualize the trajectories

int main(int argc, char** argv)
{

	std::ofstream fout;
	fout.open(argv[2], std::ios::out|std::ios::binary|std::ios::trunc);

	if (!fout) {
		std::cout << "file.txt cannot be open"<<std::endl;
		return 1;
	}



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
	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo, video); 
	std::vector<Frame> bb_list;
	
	if(bb_file) {
		LoadBoundBox(bb_file, bb_list);
		assert(bb_list.size() == seqInfo.length);
	}

	if(flag)
		seqInfo.length = end_frame - start_frame + 1;

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

	std::vector<float> fscales(0);	
	std::vector<Size> sizes(0);	

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

			for(int iScale = 0; iScale < scale_num; iScale++) {
				if(iScale == 0)
					prev_grey.copyTo(prev_grey_pyr[0]);
				else
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

				std::vector<Point2f> points(0);
				DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);

				// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];


				for(i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
				
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

			std::list<Track>& tracks = xyScaleTracks[iScale];
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
				int index = iTrack->index;

				Point2f prev_point = iTrack->point[index];

				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];

				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);
					continue;
				}

				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];

				// get the descriptors for the feature point
				RectInfo rect;

				iTrack->addPoint(point);

				if(show_track == 1 && iScale == 0)
					DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);

				if(iTrack->index >= trackInfo.length) {

					std::vector<Point2f> trajectory(trackInfo.length+1);
					for(int i = 0; i <= trackInfo.length; ++i){
						trajectory[i] = iTrack->point[i]*fscales[iScale];
					}


					std::vector<Point2f> displacement(trackInfo.length);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i]*fscales[iScale];		

					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
					if(IsValid(trajectory,trajtrack, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement)) {

						// output the trajectory////////////////////////
						for (int i = 0; i < trackInfo.length; ++i){
							traLen[PointNumber].push_back(displacement[i]);
							traPoint[PointNumber].push_back(trajtrack[i]);
						}
						PointNumber ++;
					}

					iTrack = tracks.erase(iTrack);
					continue;
				}
				++iTrack;
			}


			if(init_counter != trackInfo.gap)
				continue;

			// detect new feature points every gap frames
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


		if(frame_num > 14 && PointNumber > 0){

			int width = grey_pyr[0].cols;
			int height = grey_pyr[0].rows;

			std::vector<Point2f> all_point(PointNumber);
			std::vector<vector<vector<int> > > pointINcell(width, vector<vector<int> >(height));

			for(int l = 0; l < PointNumber; l++){ 
				all_point[l].x = std::min<int>(std::max<int>(cvFloor(traPoint[l][0].x/CELL_SIZE), 0), width-1);
				all_point[l].y = std::min<int>(std::max<int>(cvFloor(traPoint[l][0].y/CELL_SIZE), 0), height-1);
				pointINcell[all_point[l].x][all_point[l].y].push_back(l);
			}


			for(int y = 0; y < ((height/CELL_SIZE) - BLOCK_SIZE + 1); y++){
				for(int x = 0; x < ((width/CELL_SIZE) - BLOCK_SIZE +1 ); x++){
					int NoneCount = 0;
					int cellnumber = 0;

					for(j = 0; j < BLOCK_SIZE; j++){
						for(i = 0; i < BLOCK_SIZE; i++){
							std::vector<Point2f> cell (track_length);
							std::vector<float> cellmean (track_length*25);

							if(pointINcell[x+i][y+j].size() > 0){
								if(NoneCount > 0){
									for(int m = 0; m < track_length * NoneCount; m++){
										cellmean[m] = 0;
										fout.write(( char * ) &cellmean[m],sizeof( float ) );
										fout.write(( char * ) &cellmean[m],sizeof( float ) );
									}
									NoneCount = 0;
								}
								for(int m = 0; m < track_length; m++){
									for(int l = 0; l < pointINcell[x+i][y+j].size();l++){
										cell[m].x  += traLen [pointINcell[x+i][y+j][l]] [m].x ;/**250;*/
										cell[m].y  += traLen [pointINcell[x+i][y+j][l]] [m].y ;/**250;*/
									}
								} 
								for(int m = 0; m < track_length; m++){
									cellmean[m*2] = cell[m].x / pointINcell[x+i][y+j].size();
									cellmean[m*2 +1] = cell[m].y / pointINcell[x+i][y+j].size();

									fout.write(( char * ) &cellmean[m*2], sizeof( float ) );
									fout.write(( char * ) &cellmean[m*2+1], sizeof( float ) );

								}
							}

							else{
								if(NoneCount == cellnumber){ 
									NoneCount++;
								}
								else{
									for(int m = 0; m < track_length * 2; m++){
										cellmean[m] = 0;

										fout.write(( char * ) &cellmean[m],sizeof( float ) );
									}
								}
							}cellnumber++;
						}
					}
				}
			}
		}

		prev_kpts_surf = kpts_surf;
		desc_surf.copyTo(prev_desc_surf);

		frame_num++;

		if( show_track == 1 ) {
			imshow( "DenseTrackStab", image);
			c = cvWaitKey(3);
			if((char)c == 27) break;
		}
	}

	clock_t end = clock();
	std::cout << "duration = " << (double)(end - start) / CLOCKS_PER_SEC << "sec.\n";

	if( show_track == 1 )
	destroyWindow("DenseTrackStab");

	fout.close(); 
	return 0;
}
