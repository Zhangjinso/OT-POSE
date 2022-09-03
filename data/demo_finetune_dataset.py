
from data.demo_dataset import DemoDataset

class DemoFinetuneDataset(DemoDataset):
    def __init__(self, data_root, opt, load_from_dataset=False):
        super(DemoFinetuneDataset, self).__init__(data_root, opt, load_from_dataset)    

    def load_item(self,  reference_img_path, label_path=None):
        reference_img_path = self.transfrom_2_demo_path(reference_img_path)
        label_path = self.transfrom_2_demo_path(label_path)
        s_label_path = self.transfrom_2_demo_path('/'.join(reference_img_path.split('/')[1:])[:-4]+'.txt')

        # label_path = self.img_to_label(label_path)   
        reference_img = self.get_image_tensor(reference_img_path)[None,:]


        label, face_center = self.get_label_tensor(label_path.strip())
        s_label, s_face_center = self.get_label_tensor(s_label_path)

        return {'reference_image':reference_img, 
                'source_skeleton':s_label[None, :], 
                'target_skeleton':label[None,:], 
                'face_center':s_face_center[None,:],
                }