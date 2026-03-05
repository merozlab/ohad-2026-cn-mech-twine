import numpy as np

# -----------tracking file readers---------------
def funcget_tracked_data(filename,obj=0,camera='nikon'):
    with open(filename,"r") as datafile:
        lines= datafile.readlines()
        # del lines[0] # remove first line to avoid 2 zero times
        N=np.size(lines,0) # number of lines
        xtl=[[]]*N # x top left
        ytl=[[]]*N # y top left
        w=[[]]*N # box width
        h=[[]]*N # box height
        indx=[[]]*N # tracked object indx
        xcntr=[[]]*N # box x center
        ycntr=[[]]*N # box y center
        # dist=[[]]*N # distance from equilibrium position
        timer=[[]]*N # time marks
        timer[0]=0 # time starts at zero
        if camera=='nikon':
            timer = [30*x for x in range(N)] # check
        # timer_epoch1=[[]]*N
        i=0 # count rows
    # x,y,w,h ; start at upper left corner
        for line in lines:
            if line==[]: break
            currentline = line.split(",") # split by ','
            indx[i]=int(currentline[-2]) # get indx of tracked object
            if indx[i] in obj:        # if current line belongs to the requested tracked object
                xtl[i]=float(currentline[0]) # xtl- x top left
                ytl[i]=float(currentline[1])
                w[i]=float(currentline[2])
                h[i]=float(currentline[3])
                xcntr[i]=xtl[i]+w[i]/2 # calculate x coordinate of box center
                ycntr[i]=ytl[i]-h[i]/2
                # if view=='top':
                #     dist[i]=np.sqrt((xcntr[i]-xcntr[0])**2+(ycntr[i]-ycntr[0])**2)
                # else:
                #     dist[i]=abs(xcntr[i]-xcntr[0])
                # if camera=='nikon':
                #     timer = [30*x for x in range(N)]

            else:
                print('skipped non-selected tracked object')
                print(timer[i],type(timer[i]),i,currentline)
            i+=1

        # x = np.subtract(xcntr,xcntr[0]) # return x relative to start point
        # y = np.subtract(ycntr,ycntr[0]) # return y relative to start point
        return xcntr,ycntr,timer

def get_tracked_data(filename,obj=[0],camera='nikon'):
    '''get tracked data for 1 or more objects from tracking file'''
    with open(filename,"r") as datafile:
        lines= datafile.readlines()
        N=np.size(lines,0) # number of lines
        xtl=[[]]*N # x top left
        ytl=[[]]*N # y top left
        w=[[]]*N # box width
        h=[[]]*N # box height
        indx=[[]]*N # tracked object indx
        xcntr=[[]]*N # box x center
        ycntr=[[]]*N # box y center
        timer=[[]]*N # time marks
        timer[0]=0 # time starts at zero
        if camera=='nikon':
            timer = [30*x for x in range(N)] # check
        i=0 # count rows
    # x,y,w,h ; start at upper left corner
        for line in lines:
            if line==[]: break
            currentline = line.split(",") # split by ','
            indx[i]=int(currentline[-2]) # get indx of tracked object
            if indx[i] in obj:        # if current line belongs to the requested tracked object
                xtl[i]=float(currentline[0]) # xtl- x top left
                ytl[i]=float(currentline[1])
                w[i]=float(currentline[2])
                h[i]=float(currentline[3])
                xcntr[i]=xtl[i]+w[i]/2 # calculate x coordinate of box center
                ycntr[i]=ytl[i]-h[i]/2

            else:
                print('skipped non-selected tracked object')
                print(timer[i],type(timer[i]),i,currentline)
            i+=1

    return xcntr,ycntr,timer,indx
