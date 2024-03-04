import nibabel as nib
import vtk
import numpy as np


img = nib.load("image_lr.nii.gz")
img_data = img.get_fdata()
dims = img.shape
spacing = (img.header['pixdim'][1], img.header['pixdim']
           [2], img.header['pixdim'][3])

# Create a VTK image object
image = vtk.vtkImageData()
image.SetDimensions(dims[0], dims[1], dims[2])
image.SetSpacing(spacing[0], spacing[1], spacing[2])
image.SetOrigin(0, 0, 0)

# Configure scalar components based on VTK version
if vtk.VTK_MAJOR_VERSION <= 5:
    image.SetNumberOfScalarComponents(1)
    image.SetScalarTypeToDouble()
else:
    image.AllocateScalars(vtk.VTK_DOUBLE, 1)

# Fill in the image data
for z in range(dims[2]):
    for y in range(dims[1]):
        for x in range(dims[0]):
            scalardata = img_data[x][y][z]
            image.SetScalarComponentFromDouble(x, y, z, 0, scalardata)

# Using Marching Cubes algorithm
Extractor = vtk.vtkMarchingCubes()
Extractor.SetInputData(image)
Extractor.SetValue(0, 150)

stripper = vtk.vtkStripper()  # Form closed polylines from input lines (poly-lines)
stripper.SetInputConnection(Extractor.GetOutputPort())

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(stripper.GetOutputPort())
mapper.ScalarVisibilityOff()

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1, 1, 0.5)
actor.GetProperty().SetOpacity(0.9)
actor.GetProperty().SetAmbient(0.15)
actor.GetProperty().SetDiffuse(0.6)
actor.GetProperty().SetSpecular(0.5)


# Initialize renderer, render window, and interactor
ren = vtk.vtkRenderer()
ren.SetBackground(1, 1, 1)
ren.AddActor(actor)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(500, 500)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.Initialize()
renWin.Render()
iren.Start()
