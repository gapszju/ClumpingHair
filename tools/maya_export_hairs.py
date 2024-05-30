import maya.standalone
maya.standalone.initialize()

import os
import time
import re
import xgenm
from maya import cmds, mel


def pluginsCheck():
    ''
    try:
        cmds.loadPlugin('AbcExport')
        cmds.loadPlugin('AbcImport')
        cmds.loadPlugin('xgenToolkit')
    except Exception:
        None
        e = None
        None
        print ("--= Your plugins (AbcExport, AbcImport, xgenToolkit) didn't load")


def run(filePath, outputDir, fxModules = [], fxSelectOnly=True):
    '''Export Curves as Alembic for using in Unreal'''
    # check #########################################
    start = time.time()
    pluginsCheck()
    workspacePath = os.path.abspath(os.path.join(filePath, '../..'))
    cmds.workspace(workspacePath, openWorkspace=True)
    cmds.file(filePath, open=True, force=True)
    workspaceDir = cmds.workspace(q = True, rootDirectory = True)
    print ('--= Your Workspace:', workspaceDir)
    
    if cmds.objExists('xgGroom'):
        cmds.delete('xgGroom')
    currentFrame = cmds.currentTime(q = True)
    print ('--= Time (checking): {}'.format(time.time() - start))
    
    # convert to interactive groom #########################################
    start = time.time()
    allDesc = xgenm.descriptions()
    allDesc = [i for i in allDesc if cmds.getAttr( i+'.visibility')]
    allSplineDesc = []

    for desc in allDesc:
        # guides
        descParentColl = cmds.listRelatives(desc, parent = True)[0]
        descGroom = xgenm.getAttr('groom', str(descParentColl), desc)
        if descGroom not in cmds.listRelatives(desc):
            descShape, descSubdPatch = cmds.listRelatives(desc)
            descGuides = cmds.listRelatives(descSubdPatch, fullPath = True)
            cmds.select(descGuides, r = True)
            mel.eval('xgmCreateCurvesFromGuides 1 0;')
            cmds.rename('xgCurvesFromGuides1', desc.split('|')[-1] + '_guides')
        
        # modifiers
        if fxSelectOnly:
            pale = xgenm.palette(desc)
            for mod in xgenm.fxModules(pale, desc):
                if not re.sub(r'\d+$', '', mod) in fxModules:
                    xgenm.removeFXModule(pale, desc, mod)
        
        # convert
        cmds.select(desc, r = True)
        cmds.xgmGroomConvert()
        splineDesc = desc.split('|')[-1] + '_splineDescription'
        allSplineDesc.append(splineDesc)

    print ('--= Time (converting): {}'.format(time.time() - start))
    
    # export to alembic file #########################################
    os.makedirs(outputDir, exist_ok=True)
    
    start = time.time()
    for splineDesc in allSplineDesc:
        curveExportPath = os.path.join(outputDir, splineDesc.replace(":", "_") + '.abc')
        cmds.xgmSplineCache(export = True, j = f'-file {curveExportPath} -df "ogawa" -fr {str(currentFrame)} {str(currentFrame)} -step 1 -wfw -obj {splineDesc}')
    
    # conbine all curves into one group
    for child in cmds.listRelatives('xgGroom', children=True, fullPath=True):
        curveShapes = cmds.listRelatives(child, type='nurbsCurve', allDescendents=True)
        for curve in curveShapes:
            parent = cmds.listRelatives(curve, parent=True, fullPath=True)[0]
            cmds.parent(curve, child, shape=True, relative=True)
            cmds.delete(parent)
    
    guideExportPath = os.path.join(outputDir, 'guides.abc')
    job = f'-frameRange {str(currentFrame)} {str(currentFrame)} -dataFormat "ogawa" -root xgGroom -file {guideExportPath}'
    cmds.AbcExport(j = job)
    print ('--= Time (exporting): {}'.format(time.time() - start))

    print ('Complete!')


def get_latest_scene_file(scene_path):
    files = os.listdir(scene_path)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(scene_path, x)))
    for file in files[::-1]:
        if file.startswith("DD") and file.endswith((".ma", ".mb")):
            return os.path.join(scene_path, file)


if __name__ == "__main__":
    data_dir = "D:/dd_hair/DD_man"
    out_dir = "C:/Users/tangz/Desktop/research/code/hair_modeling/src/contrastive_learning/data/assets/DD_man/"
    
    for dir in os.listdir(data_dir):
        scene_path = os.path.join(data_dir, dir, "scenes")
        if os.path.isdir(scene_path):
            proj_path = get_latest_scene_file(scene_path)
            print(f"Processing {proj_path}...")

            run(proj_path, os.path.join(out_dir, dir, "Clumping"), fxModules=['Clumping'])
            run(proj_path, os.path.join(out_dir, dir, "Noise"), fxModules=['Noise'])
            run(proj_path, os.path.join(out_dir, dir, "Wo_Modifiers"), fxModules=[])
            run(proj_path, os.path.join(out_dir, dir, "Full"), fxSelectOnly=False)


maya.standalone.uninitialize()
