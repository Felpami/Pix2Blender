bl_info = {
    "name": "Pix2Blender",
    "author": "Federico Filipponi, Leonardo Lucarelli, Alessandrino Manilii",
    "version": (1, 0, 3),
    "blender": (2, 90, 0),
    "location": "3D View > Tools",
    "description": "Implementation of Pix2Vox network with Blender",
    "support": 'TESTING',
    "category": "Tools"
}

import bpy
import subprocess
import os
import time
from datetime import datetime as dt

from pathlib import Path

from bpy.props import (StringProperty,
                       PointerProperty,
                       BoolProperty,
                       FloatProperty,
                       EnumProperty
                       )

from bpy.types import (Panel,
                       Operator,
                       AddonPreferences,
                       PropertyGroup,
                       )

# ------------------------------------------------------------------------
#    Scene Properties
# ------------------------------------------------------------------------

class MyProperties(PropertyGroup):

    path_images : StringProperty(
        name="Images",
        description="Path to Images",
        default="",
        maxlen=1024,
        subtype='DIR_PATH')
    path_script : StringProperty(
        name="Script",
        description="Path to Script",
        default="",
        maxlen=1024,
        subtype='DIR_PATH')
    path_weight : StringProperty(
        name="Weight",
        description="Path to Pix2Vox Weight",
        default="",
        maxlen=1024,
        subtype='FILE_PATH')
    bool_path : BoolProperty(
        name="Enable or Disable",
        description="Show Preview",
        default = True
        ) 
    preview : BoolProperty(
        name="Enable or Disable",
        description="Show Preview",
        default = False
        )
    ext : EnumProperty(
        items=[
            ('png', '.png', 'png', '', 0),
            ('jpg', '.jpg', 'jpg', '', 1),
            ('jpeg', '.jpeg', 'jpeg', '', 2),
        ],
        default = 'png'
    )

class Extension(PropertyGroup):
    ext = EnumProperty(
        items=[
            ('png', '.png', 'png', '', 0),
            ('jpg', '.jpg', 'jpg', '', 1),
            ('jpeg', '.jpeg', 'jpeg', '', 2),
        ],
        default='png'
        )

class CreateOperator(Operator):
    """Print object name in Console"""
    bl_idname = "object.simple_operator"
    bl_label = "Simple Object Operator"
    
    def execute(self, context):
        
        ex = bpy.context.scene.my_tool.ext
        bo = bpy.context.scene.my_tool.bool_path
        pre = bpy.context.scene.my_tool.preview
        p_im = bpy.context.scene.my_tool.path_images
        p_sc = bpy.context.scene.my_tool.path_script
        p_we = bpy.context.scene.my_tool.path_weight
        #ret = 0 #KONOPEPPAPA

        pybin = bpy.app.binary_path_python

        os.system('cls')

        bpy.ops.wm.console_toggle()

        try:
            print('[INFO] %s Upgrading pip.' % (dt.now()))
            subprocess.check_call([pybin, '-m', 'pip', 'install', '--upgrade', 'pip', '--user'])
            print('[INFO] %s Successfully upgraded pip.' % (dt.now()))
        except:
            print('[INFO] %s Cannot upgrade pip, continuing execution with pip older version.' % (dt.now()))
            pass

        if(bo):
            try:
                p_sc = str(Path(os.path.realpath(__file__)).parent) + '\\core\\'
                subprocess.check_call([pybin,'Pix2Vox.py','--image_folder_path=%s' % (p_im), '--plot=%s' % (pre), '--extension=%s' % (ex)], cwd=p_sc)
            except:
                time.sleep(2)
                bpy.ops.wm.console_toggle()
                return {'FINISHED'}
        else:
            try:
                subprocess.check_call([pybin,'Pix2Vox.py','--weight=%s' % (p_we),'--image_folder_path=%s' % (p_im), '--plot=%s' % (pre), '--extension=%s' % (ex)], shell=True, cwd=p_sc)
            except:
                time.sleep(2)
                bpy.ops.wm.console_toggle()
                return {'FINISHED'}
        
        bpy.ops.import_scene.vox(filepath=p_sc + '\\vox_output\\Vox.vox')
        vox_object = bpy.context.selected_objects[0] ####<--Fix
        bpy.context.view_layer.objects.active = vox_object
        print('[INFO] %s Joining all object.' % (dt.now()))
        bpy.ops.object.join()
        obj = bpy.context.object
        bpy.ops.object.editmode_toggle()
        print('[INFO] %s Removing all doubles vertex.' % (dt.now()))
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.editmode_toggle()
        print('[INFO] %s Deleting the materials.' % (dt.now()))
        obj.data.materials.clear()
        print('[INFO] %s Applying Remehs modfifier.' % (dt.now()))
        bpy.ops.object.modifier_add(type='REMESH')
        bpy.context.object.modifiers["Remesh"].voxel_size = 1
        print('[INFO] %s Applying Subdivion Surface modfifier.' % (dt.now()))
        bpy.ops.object.modifier_add(type='SUBSURF')
        print('[INFO] %s Finished.' % (dt.now()))
        time.sleep(2)
        bpy.ops.wm.console_toggle()

        return {'FINISHED'}
    
class ApplyOperator(Operator):
    bl_idname = "object.apply_operator"
    bl_label = "Apply operator"
        
    def execute(self, context):
        obj = context.object
        if ("Remesh" in obj.modifiers):
            bpy.ops.object.modifier_apply(modifier="Remesh",report=True)
        if ("Subdivision" in obj.modifiers):
            bpy.ops.object.modifier_apply(modifier="Subdivision", report=True)
        
        return {'FINISHED'}

class BoolOperator(Operator):
    bl_idname = "object.bool_operator"
    bl_label = "Bool operator"
        
    def execute(self, context):
        return {'FINISHED'}
# ------------------------------------------------------------------------
#    Panel in Object Mode
# ------------------------------------------------------------------------

class PPanel(Panel):
    bl_idname = "PPanel"
    bl_label = "Pix2Blender"
    bl_space_type = "VIEW_3D"   
    bl_region_type = "UI"
    bl_category = "Tools"
    bl_context = "objectmode"

    def draw(self, context):
        layout = self.layout
        scn = context.scene
        obj = context.object
        layout.prop(scn.my_tool, "bool_path", text="Use default path and weights")
        layout.prop(scn.my_tool, "ext", expand=True)
        
        row = layout.row()
        row = layout.row()
        
        layout.prop(scn.my_tool, "path_images", text="Images Folder")
        if (not bpy.context.scene.my_tool.bool_path):
            layout.prop(scn.my_tool, "path_script", text="Script Folder")
            layout.prop(scn.my_tool, "path_weight", text="Weight File")
            
        row = layout.row()
        
        layout.prop(scn.my_tool, "preview", text="Preview Volume")
        if (bpy.context.scene.my_tool.bool_path):
            if (bpy.context.scene.my_tool.path_images != ""):
                layout.operator(CreateOperator.bl_idname, text="Create", icon="CUBE")
        else:
            if ((bpy.context.scene.my_tool.path_images != "") and (bpy.context.scene.my_tool.path_script != "") and (bpy.context.scene.my_tool.path_weight != "")):
                layout.operator(CreateOperator.bl_idname, text="Create", icon="CUBE")

        row = layout.row()

        #if "Remesh" in obj.modifier":

        if obj is not None and obj.modifiers:
            if ("Remesh" in obj.modifiers):
                mod = obj.modifiers["Remesh"]
                #mod.voxel_size = 1.0
                layout.label(text="Remesh Modifiers Property:")
                layout.prop(mod, "voxel_size")
                layout.prop(mod, "adaptivity")
                layout.prop(mod, "use_smooth_shade")

            row = layout.row()

            if ("Subdivision" in obj.modifiers):
                mod1 = obj.modifiers["Subdivision"]
                layout.label(text="Subdivision Modifiers Property:")
                layout.prop(mod1, "levels")
                layout.prop(mod1, "render_levels")
                layout.prop(mod1, "quality")

            row = layout.row()

            layout.operator(ApplyOperator.bl_idname, text="Apply", icon="CUBE")

            row = layout.row()

        else:
            if (bpy.context.scene.my_tool.path_images == ""):
                layout.label(text="Select an image folder.")
            if (not bpy.context.scene.my_tool.bool_path):
                if (bpy.context.scene.my_tool.path_script == ""):
                    layout.label(text="Select the script main folder.")
                if (bpy.context.scene.my_tool.path_weight == ""):
                    layout.label(text="Select the weights file.")

# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------

classes = (
    MyProperties,
    CreateOperator,
    ApplyOperator,
    BoolOperator,
    PPanel
)

def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    bpy.types.Scene.my_tool = PointerProperty(type=MyProperties)

def unregister():
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)
    del bpy.types.Scene.my_tool


if __name__ == "__main__":
    register()