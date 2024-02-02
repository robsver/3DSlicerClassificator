import logging
import os

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import SimpleITK as sitk
import sitkUtils

try:
    import monai
    import torch
    from monai.transforms import (
        Compose,
        EnsureChannelFirst,
        ScaleIntensity,
        SpatialPad
    )
    from monai.data import DataLoader, ImageDataset
    from monai.data import MetaTensor
    import torch.nn as nn
  
except ModuleNotFoundError:
    slicer.util.pip_install("monai")
    import monai
    import torch


#
# Classificator
#

class Classificator(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Classificator"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Classification"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Roberto Veraldi (Magna Graecia University of Catanzaro, Italy)", "Paolo Zaffino (Magna Graecia University of Catanzaro, Italy)", "Maria Francesca Spadea (KIT, Germany)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#Classificator">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        # slicer.app.connect("startupCompleted()", registerSampleData)


#
# ClassificatorWidget
#

class ClassificatorWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/Classificator.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ClassificatorLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        #self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        #self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

    def onApplyButton(self, toggle):
        print("hello")
        predictedLabel = self.logic.runClassification(self.ui.inputSelector.currentNode().GetName(), self.ui.PathLineEdit.currentPath)
        self.ui.predictedLabel.setText(predictedLabel)


    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        # self.initializeParameterNode()
        pass

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    
    #def onSceneStartClose(self, caller, event):
    #    """
    #    Called just before the scene is closed.
    #    """
    #    # Parameter node will be reset, do not use it anymore
    #    self.setParameterNode(None)

    #def onSceneEndClose(self, caller, event):
    #    """
    #    Called just after the scene is closed.
    #    """
    #    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    #    if self.parent.isEntered:
    #        self.initializeParameterNode()

#
# ClassificatorLogic
#

class ClassificatorLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def unpad(self, tens, background=0):

        last_cut = 0
        for i in range(tens.shape[1]):
            if tens[:,i,:].sum() != background:
                last_cut = i

        first_cut = 0
        for i in range(tens.shape[1]-1,-1,-1):
            if tens[:,i,:].sum() != background:
                first_cut = i
        
        return tens[:,first_cut:last_cut,:]

    def slices2d(self, tensor3d: torch.tensor) -> torch.tensor:

        new_tensor = torch.zeros((tensor3d.shape[0], 5, tensor3d.shape[2], tensor3d.shape[4]), dtype=torch.float32)

        for i in range(tensor3d.shape[0]):
            unpad_tensor = self.unpad(tensor3d[i,0,:,:,:])
            ending_slice = unpad_tensor.shape[1]
            index_central = int(ending_slice*0.5)
            index_upper1 = int(ending_slice*0.7)
            index_upper2 = int(ending_slice*0.8)
            index_lower1 = int(ending_slice*0.3)
            index_lower2 = int(ending_slice*0.2)

            new_tensor[i,0,:,:]=unpad_tensor[:,index_upper2,:]
            new_tensor[i,1,:,:]=unpad_tensor[:,index_upper1,:]
            new_tensor[i,2,:,:]=unpad_tensor[:,index_central,:]
            new_tensor[i,3,:,:]=unpad_tensor[:,index_lower1,:]
            new_tensor[i,4,:,:]=unpad_tensor[:,index_lower2,:]
        
        return new_tensor

    def runClassification(self, VolumeNodeName, weight_fn):
        volume_sitk = sitk.Cast(sitkUtils.PullVolumeFromSlicer(VolumeNodeName), sitk.sitkFloat32)
        print(volume_sitk.GetSize())
        # I need VolumeNodeName path 
        # n = slicer.util.getFirstNodeByName("patient_crop03_515")
        # n.GetStorageNode().GetFullNameFromFileName()
        print(f"I'm going to classify {VolumeNodeName} by using {weight_fn}")
        #print(volume_name)
        
        # volume = sitk.ReadImage(volume_sitk)
        volume_np = sitk.GetArrayFromImage(volume_sitk)
        volume_torch = torch.tensor(volume_np.T)
        volume_torch_5d = torch.reshape(volume_torch, (1,1,volume_torch.shape[0],volume_torch.shape[1],volume_torch.shape[2]))

        transformation = Compose([
                        SpatialPad(spatial_size=(1,134,49,81), value=-1000),
                        ScaleIntensity(minv=0.0, maxv=1.0)
                        ])
        
        # Torch model
        
        model = torch.load(str(weight_fn))

        model.eval()

        with torch.inference_mode():

            inputs = transformation(volume_torch_5d)
            print(inputs.shape)
            inputs2d = self.slices2d(inputs).to("cpu")
                
            pred = model(inputs2d).to("cpu")
            print(pred)

        return int(torch.argmax(pred,1))


#
# ClassificatorTest
#

class ClassificatorTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_Classificator1()

    def test_Classificator1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('Classify1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = ClassificatorLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
