// check if you have slices or frames defined, if z-stack use slices if time series use frames to create a new stack
// load ROIs in roi manager, make sure the first number in the name is the slice/frame number
// this script does not work with multiple channels
getDimensions(width, height, channels, slices, frames)
print("width: "+width)
print("height: "+height)
print("channels: "+channels)
print("slices: "+slices)
print("frames: "+frames)


newImage("Labeling", "16-bit black", height, width, slices);

slice0 = 0;
for (index = 0; index < roiManager("count"); index++) {
	roiManager("select", index);
	rName = Roi.getName(); 
	s = split(rName,"-");
	slice1 = parseFloat(s[0]);
	setColor(index);
	fill();
}
run("Select None");
resetMinAndMax(); // applies to last image now, other may have more labels
run("hsvbr"); // whatever LUT you like

