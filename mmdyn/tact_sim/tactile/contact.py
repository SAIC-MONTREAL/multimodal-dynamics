import numpy as np
import pybullet as p


class Contact:
    """Contact
    Parse and hold the contact information."""

    def __init__(self, base_body_id):
        """
        Each point in contact_points has:
                contactFlag (int)           : reserved
                bodyUniqueIdA (int)         : body unique id of body A
                bodyUniqueIdB (int)         : body unique id of body B
                linkIndexA (int)            : link index of body A, -1 for base
                linkIndexB (int)            : link index of body B, -1 for base
                positionOnA (list)          : contact position on A, in Cartesian world coordinates
                positionOnB (list)          : contact position on B, in Cartesian world coordinates
                contactNormalOnB (list)     : contact normal on B, pointing towards A
                contactDistance (float)     : contact distance, positive for separation, negative for penetration
                normalForce (float)         : normal force applied during the last 'stepSimulation'
                lateralFriction1 (float)    : lateral friction force in the lateralFrictionDir1 direction
                lateralFrictionDir1 (list)  : first lateral friction direction
                lateralFriction2 (float)    : lateral friction force in the lateralFrictionDir2 direction
                lateralFrictionDir2 (list)  : second lateral friction direction
        Some of the values are dropped as they are not useful to us now, but can be easily added as we see fit.
        Args:
            base_body_id (int)              : Unique ID of the base body to obtain its contacts.
        """
        self._contacts = p.getContactPoints(base_body_id)
        self._body_ids = []
        self._points = []
        self._normal_forces = []

        for contact in self._contacts:
            # drop contacts with the ground
            if contact[2] != 0:
                self._body_ids.append(contact[2])
                self._points.append(list(contact[5]))
                self._normal_forces.append(contact[9])

    def __len__(self):
        return len(self._body_ids)

    def total_force(self, body_id):
        """
        Returns the total force applied on the body with the specified id.
        Args:
            body_id (int)           : ID of the body in question.

        Returns:
            float                   : Total force applied to the body.
        """
        return self.info['normal_forces'][np.where(self.info['body_ids'] == body_id)].sum()

    @property
    def unique_ids(self):
        """
        A list of all body IDs in contact.
        """
        return list(set(self._body_ids))

    @property
    def info(self):
        return {
            'body_ids': np.reshape(self._body_ids, (-1, 1)),
            'points': np.reshape(self._points, (-1, 3)),
            'normal_forces': np.reshape(self._normal_forces, (-1, 1))
        }

