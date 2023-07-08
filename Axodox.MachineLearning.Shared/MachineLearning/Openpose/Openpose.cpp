#include "pch.h"
#include "Openpose.h"

void Openpose::detect(const float *input_ptr, const float *paf_ptr, Axodox::Graphics::TextureData& frame)
{
	/*
	 Input arguments:
		cmap: feature maps of joints
		paf: connections between joints
		frame: image data

	 output arguments:
		object_counts_ptr[0];			// N
		objects_ptr[0];					// NxMxC
		refined_peaks_ptr[0];			// NxCxMx2
	*/

	// ****** DETECT SKELETON ***** //

	// 1. Find peaks (NMS)

	size_t peak_size = N * C * M * 2;	// NxCxMx2
	int *peaks = new int[peak_size];

	size_t peak_count_size = N * C; // NxC
	int *peak_counts = new int[peak_count_size];

	trt_pose::parse::find_peaks_out_nchw(peak_counts, peaks, input_ptr, N, C, H, W, M, cmap_threshold, cmap_window);

	// 2. Refine peaks

	float *refined_peaks = new float[peak_size]; // NxCxMx2

	for (int i = 0; i < peak_size; i++) refined_peaks[0] = 0;

	trt_pose::parse::refine_peaks_out_nchw(refined_peaks, peak_counts, peaks, input_ptr, N, C, H, W, M, cmap_window);

	// 3. Score paf 
 
	int K = 21;

	size_t score_graph_size = N * K * M * M;	// NxKxMxM
	float *score_graph = new float[score_graph_size];

	trt_pose::parse::paf_score_graph_out_nkhw(score_graph, topology, paf_ptr, peak_counts, refined_peaks,
		N, K, C, H, W, M, line_integral_samples);

	// 4. Assignment algorithm

	int *connections = new int[N * K * 2 * M];
	int connection_size = N * K * 2 * M;
	for (int i = 0; i < connection_size; i++) connections[i] = -1.0;

	void *workspace = (void *)malloc(trt_pose::parse::assignment_out_workspace(M));

	trt_pose::parse::assignment_out_nk(connections, score_graph, topology, peak_counts, N, C, K, M, link_threshold, workspace);

	// 5. Merging

	int *objects = new int[N * max_num_objects * C];
	for (int i = 0; i < N * max_num_objects * C; i++) objects[i] = -1;

	int *object_counts = new int[N];
	object_counts[0] = 0;	// batchSize=1		

	void *merge_workspace = malloc(trt_pose::parse::connect_parts_out_workspace(C, M));

	trt_pose::parse::connect_parts_out_batch(object_counts, objects, connections, topology, peak_counts, N, K, C, M, max_num_objects, merge_workspace);

	// ****** DRAWING SKELETON ***** //

	for (int i = 0; i < object_counts[0]; i++) {

		int *obj = &objects[C * i];

		for (int j = 0; j < C; j++) {

			int k = (int)obj[j];
			if (k >= 0) {
				float *peak = &refined_peaks[j * M * 2];
				int x = (int)(peak[k * 2 + 1] * frame.Width);
				int y = (int)(peak[k * 2] * frame.Height);
				//circle(frame, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), cv::FILLED, 4, 0);
				peak = NULL;
			}
		}

		for (int k = 0; k < K; k++) {
			int c_a = topology[k * 4 + 2];
			int c_b = topology[k * 4 + 3];

			if (obj[c_a] >= 0 && obj[c_b] >= 0) {
				float *peak0 = &refined_peaks[c_a * M * 2];
				float *peak1 = &refined_peaks[c_b * M * 2];

				int x0 = (int)(peak0[(int)obj[c_a] * 2 + 1] * frame.Width);
				int y0 = (int)(peak0[(int)obj[c_a] * 2] * frame.Height);
				int x1 = (int)(peak1[(int)obj[c_b] * 2 + 1] * frame.Width);
				int y1 = (int)(peak1[(int)obj[c_b] * 2] * frame.Height);
				////line(frame, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 2, 1);

				//if ((c_a == 5 && c_b == 7) || (c_a == 7 && c_b == 9))
				//{
				//	arrowedLine(frame, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 0, 255), 2, 1); // red				
				//}
				//else if ((c_a == 6 && c_b == 8) || (c_a == 8 && c_b == 10))
				//{
				//	arrowedLine(frame, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(255, 0, 0), 2, 1); // blue
				//}
				//else {
				//	if ((c_b == 11 && c_a == 13) || (c_b == 13 && c_a == 15) || (c_b == 12 && c_a == 14) || (c_b == 14 && c_a == 16))
				//	{
				//		line(frame, cv::Point(x1, y1), cv::Point(x0, y0), cv::Scalar(0, 255, 0), 2, 1); // green
				//	}
				//	else
				//	{
				//		line(frame, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 2, 1); // green
				//	}
				//}

				////cv::putText(frame, std::to_string(c_a), cv::Point(x0, y0), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 2, false);
				////cv::putText(frame, std::to_string(c_b), cv::Point(x1, y1), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 2, false);
			}
		}
		obj = NULL;
	}

	delete[] peaks;
	peaks = NULL;
	delete[] peak_counts;
	peak_counts = NULL;

	delete[] refined_peaks;
	refined_peaks = NULL;

	paf_ptr = NULL;
	delete[] score_graph;
	score_graph = NULL;

	delete[] connections;
	connections = NULL;

	delete[] objects;
	objects = NULL;

	delete[] object_counts;
	object_counts = NULL;

	std::free(workspace);
	std::free(merge_workspace);

}
