import math


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.drone_points = {}
        self.bird_points = {}
        self.airplane_points = {}
        self.helicopter_points = {}
        self.balloon_points = {}
        self.military_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.drone_count = 1
        self.bird_count = 1
        self.airplane_count = 1
        self.helicopter_count = 1
        self.balloon_count = 1 
        self.military_count = 1
      
   
 
    def update(self, object_name, objects_rect):      
        label_name = object_name 
        xyhw = objects_rect 
        
        
        
        new_id = None
        
        if label_name == "bird":
            new_id = self.bird_update(xyhw)
            
        if label_name == "drone":
            new_id = self.drone_update(xyhw)
            
        if label_name == "airplane":
            new_id = self.airplane_update(xyhw)
            
        if label_name == "helicopter":
            new_id = self.helicopter_update(xyhw)
            
        if label_name == "balloon":
            new_id = self.balloon_update(xyhw)
        
        if label_name == "military drone":
            new_id = self.military_update(xyhw)
        
        new_id = label_name + str(new_id[0][4])
        
        return new_id
    
    def bird_update(self, objects_rect): # object_name,
                # Objects boxes and ids
                objects_bbs_ids = []
                
                # Get center point of new object
                for rect in objects_rect:
                    x, y, w, h = rect
                    cx = (x + x + w) // 2
                    cy = (y + y + h) // 2

                    # Find out if that object was detected already
                    same_object_detected = False
                    for id, pt in self.bird_points.items():
                        dist = math.hypot(cx - pt[0], cy - pt[1])

                        if dist < 50:
                            self.bird_points[id] = (cx, cy)
                            objects_bbs_ids.append([x, y, w, h, id])
                            same_object_detected = True
                            break

                    # New object is detected we assign the ID to that object
                    if same_object_detected is False:
                        self.bird_points[self.bird_count] = (cx, cy)
                        objects_bbs_ids.append([x, y, w, h, self.bird_count])
                        self.bird_count += 1

                # Clean the dictionary by center points to remove IDS not used anymore
                new_bird_points = {}
                for obj_bb_id in objects_bbs_ids:
                    _, _, _, _, object_id = obj_bb_id
                    center = self.bird_points[object_id]
                    new_bird_points[object_id] = center

                # Update dictionary with IDs not used removed
                self.bird_points = new_bird_points.copy()
                return objects_bbs_ids  
            
    def drone_update(self,  objects_rect): 
            # Objects boxes and ids
            objects_bbs_ids = []
            
            # Get center point of new object
            for rect in objects_rect:
                x, y, w, h = rect
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2

                # Find out if that object was detected already
                same_object_detected = False
                for id, pt in self.drone_points.items():
                    dist = math.hypot(cx - pt[0], cy - pt[1])

                    if dist < 30:
                        self.drone_points[id] = (cx, cy)
                        objects_bbs_ids.append([x, y, w, h, id])
                        same_object_detected = True
                        break

                # New object is detected we assign the ID to that object
                if same_object_detected is False:
                    self.drone_points[self.drone_count] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, self.drone_count])
                    self.drone_count += 1

            # Clean the dictionary by center points to remove IDS not used anymore
            new_drone_points = {}
            for obj_bb_id in objects_bbs_ids:
                _, _, _, _, object_id = obj_bb_id
                center = self.drone_points[object_id]
                new_drone_points[object_id] = center

            # Update dictionary with IDs not used removed
            self.drone_points = new_drone_points.copy()
            return objects_bbs_ids

    def airplane_update(self,  objects_rect): 
            # Objects boxes and ids
            objects_bbs_ids = []
            
            # Get center point of new object
            for rect in objects_rect:
                x, y, w, h = rect
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2

                # Find out if that object was detected already
                same_object_detected = False
                for id, pt in self.airplane_points.items():
                    dist = math.hypot(cx - pt[0], cy - pt[1])

                    if dist < 25:
                        self.airplane_points[id] = (cx, cy)
                        objects_bbs_ids.append([x, y, w, h, id])
                        same_object_detected = True
                        break

                # New object is detected we assign the ID to that object
                if same_object_detected is False:
                    self.airplane_points[self.airplane_count] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, self.airplane_count])
                    self.airplane_count += 1

            # Clean the dictionary by center points to remove IDS not used anymore
            new_airplane_points = {}
            for obj_bb_id in objects_bbs_ids:
                _, _, _, _, object_id = obj_bb_id
                center = self.airplane_points[object_id]
                new_airplane_points[object_id] = center

            # Update dictionary with IDs not used removed
            self.airplane_points = new_airplane_points.copy()
            return objects_bbs_ids

    def helicopter_update(self, objects_rect): 
            # Objects boxes and ids
            objects_bbs_ids = []
            
            # Get center point of new object
            for rect in objects_rect:
                x, y, w, h = rect
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2

                # Find out if that object was detected already
                same_object_detected = False
                for id, pt in self.helicopter_points.items():
                    dist = math.hypot(cx - pt[0], cy - pt[1])

                    if dist < 25:
                        self.helicopter_points[id] = (cx, cy)
                        objects_bbs_ids.append([x, y, w, h, id])
                        same_object_detected = True
                        break

                # New object is detected we assign the ID to that object
                if same_object_detected is False:
                    self.helicopter_points[self.airplane_count] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, self.airplane_count])
                    self.airplane_count += 1

            # Clean the dictionary by center points to remove IDS not used anymore
            new_helicopter_points = {}
            for obj_bb_id in objects_bbs_ids:
                _, _, _, _, object_id = obj_bb_id
                center = self.helicopter_points[object_id]
                new_helicopter_points[object_id] = center

            # Update dictionary with IDs not used removed
            self.helicopter_points = new_helicopter_points.copy()
            return objects_bbs_ids

    def balloon_update(self,  objects_rect): 
            # Objects boxes and ids
            objects_bbs_ids = []
            
            # Get center point of new object
            for rect in objects_rect:
                x, y, w, h = rect
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2

                # Find out if that object was detected already
                same_object_detected = False
                for id, pt in self.balloon_points.items():
                    dist = math.hypot(cx - pt[0], cy - pt[1])

                    if dist < 25:
                        self.balloon_points[id] = (cx, cy)
                        objects_bbs_ids.append([x, y, w, h, id])
                        same_object_detected = True
                        break

                # New object is detected we assign the ID to that object
                if same_object_detected is False:
                    self.balloon_points[self.balloon_count] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, self.balloon_count])
                    self.balloon_count += 1

            # Clean the dictionary by center points to remove IDS not used anymore
            new_balloon_points = {}
            for obj_bb_id in objects_bbs_ids:
                _, _, _, _, object_id = obj_bb_id
                center = self.balloon_points[object_id]
                new_balloon_points[object_id] = center

            # Update dictionary with IDs not used removed
            self.balloon_points = new_balloon_points.copy()
            return objects_bbs_ids
        
    def military_update(self,objects_rect): 
            # Objects boxes and ids
            objects_bbs_ids = []
            
            # Get center point of new object
            for rect in objects_rect:
                x, y, w, h = rect
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2

                # Find out if that object was detected already
                same_object_detected = False
                for id, pt in self.military_points.items():
                    dist = math.hypot(cx - pt[0], cy - pt[1])

                    if dist < 25:
                        self.military_points[id] = (cx, cy)
                        objects_bbs_ids.append([x, y, w, h, id])
                        same_object_detected = True
                        break

                # New object is detected we assign the ID to that object
                if same_object_detected is False:
                    self.military_points[self.military_count] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, self.military_count])
                    self.military_count += 1

            # Clean the dictionary by center points to remove IDS not used anymore
            new_military_points = {}
            for obj_bb_id in objects_bbs_ids:
                _, _, _, _, object_id = obj_bb_id
                center = self.military_points[object_id]
                new_military_points[object_id] = center

            # Update dictionary with IDs not used removed
            self.military_points = new_military_points.copy()
            return objects_bbs_ids  
     
        
            
        
            
        
    # Original Function
    # def update(self,  objects_rect): # object_name,
    #     # Objects boxes and ids
    #     objects_bbs_ids = []
        
    #     # Get center point of new object
    #     for rect in objects_rect:
    #         x, y, w, h = rect
    #         cx = (x + x + w) // 2
    #         cy = (y + y + h) // 2

    #         # Find out if that object was detected already
    #         same_object_detected = False
    #         for id, pt in self.center_points.items():
    #             dist = math.hypot(cx - pt[0], cy - pt[1])

    #             if dist < 25:
    #                 self.center_points[id] = (cx, cy)
    #                 print(self.center_points)
    #                 objects_bbs_ids.append([x, y, w, h, id])
    #                 same_object_detected = True
    #                 break

    #         # New object is detected we assign the ID to that object
    #         if same_object_detected is False:
    #             self.center_points[self.id_count] = (cx, cy)
    #             objects_bbs_ids.append([x, y, w, h, self.id_count])
    #             self.id_count += 1

    #     # Clean the dictionary by center points to remove IDS not used anymore
    #     new_center_points = {}
    #     for obj_bb_id in objects_bbs_ids:
    #         _, _, _, _, object_id = obj_bb_id
    #         center = self.center_points[object_id]
    #         new_center_points[object_id] = center

    #     # Update dictionary with IDs not used removed
    #     self.center_points = new_center_points.copy()
    #     return objects_bbs_ids

    