import nibabel as nib
import vtk

# Load the image using nibabel
img = nib.load("image_lr.nii.gz")
img_data = img.get_fdata()  
dims = img.shape
spacing = img.header.get_zooms()

# Create a VTK image object
vtk_image = vtk.vtkImageData()
vtk_image.SetDimensions(dims[0], dims[1], dims[2])  #  Set the dimensions
vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])  # Set the spacing
vtk_image.SetOrigin(0, 0, 0)

# Fill the VTK image with scalar data
int_range = (20, 500)  # Set the intensity range
max_u_short = 128  # Set the maximum value for unsigned short
vtk_image.AllocateScalars(vtk.VTK_DOUBLE, 1)
for z in range(dims[2]):
    for y in range(dims[1]):
        for x in range(dims[0]):
            scalar_data = img_data[x, y, z]
            if scalar_data < int_range[0]:
                scalar_data = int_range[0]
            if scalar_data > int_range[1]:
                scalar_data = int_range[1]
            scalar_data = max_u_short*float(scalar_data-int_range[0])/float(int_range[1]-int_range[0])  # Scale the data
            vtk_image.SetScalarComponentFromDouble(x, y, z, 0, scalar_data)  # Set the scalar value

# Create a volume property object
volume_property = vtk.vtkVolumeProperty()

# Create transfer functions for opacity
opacity_function = vtk.vtkPiecewiseFunction()
opacity_function.AddSegment(0, 0, 10, 0)
opacity_function.AddSegment(10, 0.2, 120, 0.2)
volume_property.SetScalarOpacity(opacity_function)

# Set up a color transfer function
color_function = vtk.vtkColorTransferFunction()
color_function.AddRGBSegment(0, 0, 0, 0, 20, 0.2, 0.2, 0.2)
color_function.AddRGBSegment(20, 0.1, 0.2, 0, 128, 1, 1, 0)
volume_property.SetColor(color_function)

# Set gradient opacity function
gradient_opacity = vtk.vtkPiecewiseFunction()
gradient_opacity.AddPoint(0, 0.0)
gradient_opacity.AddSegment(27, 0.3, 128, 0.4)
volume_property.SetGradientOpacity(gradient_opacity)

# Set optical properties
volume_property.SetInterpolationTypeToLinear()
volume_property.SetAmbient(1)
volume_property.SetDiffuse(0.9)
volume_property.SetSpecular(0.6)
volume_property.SetSpecularPower(10)

# Create a volume mapper
volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
volume_mapper.SetInputData(vtk_image)
volume_mapper.SetImageSampleDistance(5.0)

# Create a volume, set its mapper and property
volume = vtk.vtkVolume()
volume.SetMapper(volume_mapper)
volume.SetProperty(volume_property)

# Initialize renderer, render window, and interactor
renderer = vtk.vtkRenderer()
renderer.SetBackground(1, 1, 1)
renderer.AddVolume(volume)

# Add a light source to the renderer
light = vtk.vtkLight()
light.SetColor(0, 1, 1)
renderer.AddLight(light)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(500, 500)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
interactor.Initialize()
render_window.Render()
interactor.Start()