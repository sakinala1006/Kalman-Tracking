—FILES—
-> /Cv project 2/input/moon_frames - has the pictures for the video of moon.
			/walking_frames - has the pictures for the video of a walking man.
-> /Cv project 2/output/moon_output_frames - has the output moon pictures from WSSD.
			   /walking_output_frames - has the output walking man pictures from WSSD.
			   /moon_kalman_frames - has the output moon pictures from Kalman’s tracking.


—FUNCTIONS—
-> featureTrack is the function which carries out autocorrelation from the second image onwards.
-> KalmanTrack is the function which carries out Kalman feature tracking.
-> Feature Detection is performed with in ‘__main__’ .
-> Iterating over image files is done within the ‘__main__’ .


---NOTE---
-> The submitted file uses the video of moon as the input file. If a new file has to be given as the input, it should be specified
    in 7 places in the code.
-> Mistakes in file extensions can lead to errors. Moon frames are in the form of .png and walking frames are in the form of .jpg .


---REFERENCES---
->Numpy documentation:
	 https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html 
-> Scipy for Gaussian Filters:
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
