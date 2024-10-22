# Point Picker to Select Landmark Positions

The following application allows you to interactively pick points on a surface mesh or point cloud and is meant to provide an easy way to select landmarks for rigid alignment.

To run the application, simply type
```
python pick_landmarks.py <path-to-point-cloud>
```
This opens up a small window. Right click to select a point. 
If you want to delete the last picked point, press `d`. 
If you want to save clicked points to a `.csv` file, press `s`. 
Hit `e` to close the window.