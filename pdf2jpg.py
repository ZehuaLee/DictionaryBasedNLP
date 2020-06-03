import sys, fitz
import os
import datetime

basefolder = "sampledata"
subfolders = os.listdir(basefolder)
resultfolder = "resultfolder"
pdfpaths = []
pdffolders = []

def pyMuPDF_fitz(pdfPath, imagePath):
    startTime_pdf2img = datetime.datetime.now()
    
    print("input=", pdfPath)
    print("imagePath prefix="+imagePath)
    pdfDoc = fitz.open(pdfPath)
    counter = 1
    for pg in range(pdfDoc.pageCount):
        page = pdfDoc[pg]
        rotate = int(0)
        #zoom_x = 1.33333333 #(1.33333333-->1056x816)   (2-->1584x1224)
        #zoom_y = 1.33333333
        zoom_x = 2 #(1.33333333-->1056x816)   (2-->1584x1224)
        zoom_y = 2

        mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pix = page.getPixmap(matrix=mat, alpha=False)
        
        #if not os.path.exists(imagePath):
        #    os.makedirs(imagePath) 
        
        img = imagePath+'_'+'{:0=2}.png'.format(counter)
        print("imagePath="+img)

        pix.writePNG(img)
        counter += 1
        
    endTime_pdf2img = datetime.datetime.now()
    print('pdf2img time =',(endTime_pdf2img - startTime_pdf2img).seconds)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("- - - {} created. - - -".format(path))
    else:
        print("- - - {} exists. - - -".format(path))
for subfolder in subfolders:
    files = os.listdir(os.path.join(basefolder,subfolder))
    for singlefile in files:
        pdfpaths.append(os.path.join(basefolder,subfolder,singlefile))

pdfpaths = [e for e in pdfpaths if '.ipynb' not in e]
for i, pdf in enumerate(pdfpaths):        
    #templst = pdf.split("\/")
    #print(templst)
    #folderpath = os.path.join(resultfolder,templst[1],templst[2][:-4])
    #mkdir(folderpath)
    folderpath = resultfolder+'/'+'{:0=4}'.format(i+1)
    #mkdir(folderpath)
    pyMuPDF_fitz(pdf, folderpath)