       
# def lossfunc(box1,box2):
#     box1x1 = (box1[:,0] - box1[:,2]).reshape(-1,1)
#     box1x2 = (box1[:,0] + box1[:,2]).reshape(-1,1)
#     box1y1 = (box1[:,1] - box1[:,2]).reshape(-1,1)
#     box1y2 = (box1[:,1] + box1[:,2]).reshape(-1,1)

#     box2x1 = (box2[:,0] - box2[:,2]).reshape(-1,1)
#     box2x2 = (box2[:,0] + box2[:,2]).reshape(-1,1)
#     box2y1 = (box2[:,1] - box2[:,2]).reshape(-1,1)
#     box2y2 = (box2[:,1] + box2[:,2]).reshape(-1,1)
    
#     print(torch.cat((box1x1,box1y1,box1x2,box1y2),dim=1))

#     ret