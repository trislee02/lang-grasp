from inference.post_process import post_process_output
from utils.dataset_processing import evaluation

class GraspAccuracy:
    def __init__(self, iou_threshold=0.25, dataset=None):
        self.iou_threshold = iou_threshold
        self.results = {
                'correct': 0,
                'failed': 0
            }
        self.dataset = dataset
        
    def update(self, lossd, didx, rot, zoom_factor):
        '''
            Update the results with the current predictions
            Only works for batch size = 1
        '''
        # TODO: Reimplement this function for batch size > 1
        q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                    lossd['pred']['sin'], lossd['pred']['width'])

        s = evaluation.calculate_iou_match(q_out,
                                            ang_out,
                                            self.dataset.get_gtbb(didx, rot, zoom_factor),
                                            no_grasps=1,
                                            grasp_width=w_out,
                                            threshold=self.iou_threshold,
                                            )

        if s:
            self.results['correct'] += 1
        else:
            self.results['failed'] += 1

    def reset(self):
        self.results = {
                'correct': 0,
                'failed': 0
            }