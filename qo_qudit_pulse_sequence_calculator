import numpy as np
import quaternion
import operator

def isqrt(n):
    """
    Return the integer part of the square root of the input.
    (math.isqrt from Python 3.8)
    """
    n = operator.index(n)
    if n < 0:
        raise ValueError("isqrt() argument must be nonnegative")
    if n == 0:
        return 0
    c = (n.bit_length() - 1) // 2
    a = 1
    d = 0
    for s in reversed(range(c.bit_length())):
        # Loop invariant: (a-1)**2 < (n >> 2*(c - d)) < (a+1)**2
        e = d
        d = c >> s
        a = (a << d - e - 1) + (n >> 2*c - e - d + 1) // a
    return a - (a*a > n)

def phi_to_axis(phi):
    '''
    Converts an angle phi to the corresponding axis in the xy-plane
    
    Inputs:
        phi (float): counterclockwise angle from the x-axis in the xy-plane
    
    Returns:
        numpy array representing the axis in the xy-plane corresponding to angle phi
    '''
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    return np.cos(phi) * x_axis + np.sin(phi) * y_axis

def xy_decomposition(axis, angle, theta = 0):
    '''
    Decompose the given rotation into a composition of rotations about axes in the xy-plane
    
    Inputs:
        axis: the axis of the rotation to be decomposed
        angle: the angle of the rotation to be decomposed
        theta: a degree of freedom for the xy decomposition, ranges from 0 to 2*pi
    
    Returns:
        list of 2-tuples in which the first element of each 2-tuple specifies an axis of rotation
        and the second element of each 2-tuple specifies an angle of rotation
    '''
    #if axis is already in xy-plane, return sequence with original rotation
    if abs(axis[2]) < 1e-12:
        rotation_sequence = [(axis, angle)]
    else:
        #find a perpendicular axis with large magnitude (to minimize numerical errors)
        perp_axis_1 = np.array([0, axis[2], -axis[1]])
        if np.linalg.norm(perp_axis_1) < 0.7:
            perp_axis_1 = np.array([-axis[2], 0, axis[1]])
        if np.linalg.norm(perp_axis_1) < 0.7:
            perp_axis_1 = np.array([-axis[1], axis[0], 0])
        
        perp_axis_1 = perp_axis_1 / np.linalg.norm(perp_axis_1)
        
        #find an axis perpendicular to both axes
        perp_axis_2 = np.cross(axis, perp_axis_1)
        perp_axis_2 = perp_axis_2 / np.linalg.norm(perp_axis_2)
        
        #compute the axes of reflection for the two reflections that compose to give this rotation
        reflect_1_axis = np.cos(theta) * perp_axis_1 + np.sin(theta) * perp_axis_2
        reflect_2_axis = np.cos(theta + angle / 2) * perp_axis_1 + np.sin(theta + angle / 2) * perp_axis_2
        
        #define the z-axis normal vector
        z_axis = np.array([0, 0, 1])
        
        #compose the first reflection with a reflection through the xy-plane and find the corresponding rotation
        rotate_1_axis = np.cross(reflect_1_axis, z_axis)
        rotate_1_axis = rotate_1_axis / np.linalg.norm(rotate_1_axis)
        rotate_1_angle = 2 * np.arccos(np.dot(reflect_1_axis.T, z_axis))
        
        #compose a reflection through the xy-plane with the second rotation and find the corresponding rotation
        rotate_2_axis = np.cross(z_axis, reflect_2_axis)
        rotate_2_axis = rotate_2_axis / np.linalg.norm(rotate_2_axis)
        rotate_2_angle = 2 * np.arccos(np.dot(reflect_2_axis.T, z_axis))
        
        #create a rotation sequence corresponding to these two rotations
        rotation_sequence = [(rotate_1_axis, rotate_1_angle), (rotate_2_axis, rotate_2_angle)]     
    
    return rotation_sequence

def axis_switch(init_axis, final_axis):
    '''
    Provide the axis and angle required to rotate from one vector to another
    
    Inputs:
        init_axis: the initial axis vector to be rotated
        final_axis: the target axis vector to which the initial axis vector is to be rotated
    
    Returns:
        vector and scalar representing the axis and angle of a rotation that takes init_axis to final_axis
    '''
    #compute the rotation axis that is perpendicular to both axes to be the axis of rotation
    #compute the rotation angle as the angle between the initial axis and the final axis
    rotation_axis = np.cross(init_axis, final_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation_angle = np.arccos(np.dot(init_axis.T, final_axis))
    
    return rotation_axis, rotation_angle

def alpha_beta_decomposition(subspace_angle, gate = "X"):
    '''
    Given an angle and a gate, compute the two axes of the rotations and the number of rotations to make this gate
    In particular, the gate is given by num_b_rotations rotations of angle subspace_angle about axis beta
    followed by one rotation of angle subspace_angle about angle alpha
    
    Inputs:
        subspace_angle: the angle of rotation
        gate: the gate to be obtained
    
    Returns:
        two axes alpha and beta and a number of rotations num_b_rotations
        such that num_b_rotations rotations of angle subspace_angle about axis beta
        followed by one rotation of angle subspace_angle about angle alpha
        results in the desired gate
    '''
    
    #calculate the number of rotations required to achieve an X or Y gate (i.e., a rotation by pi)
    effective_rot_angle = subspace_angle if subspace_angle <= np.pi else 2 * np.pi - subspace_angle
    num_b_rotations = int(np.ceil(np.pi / effective_rot_angle)) - 1
    
    #define two convenient angles for calculation of alpha and beta
    angle1 = subspace_angle / 2
    angle2 = num_b_rotations * subspace_angle / 2
    
    #solve for the components of the axes for each gate type
    if gate == "X":
        #b2 is a free parameter that must have magnitude at most sqrt(1 - cos^2(angle1)/sin^2(angle2))
        b2 = 0
        b1 = np.cos(angle1) / np.sin(angle2)
        b3 = 1 / np.sin(angle2) * np.sqrt(-1 * np.cos(angle1) ** 2 + np.sin(angle2) ** 2 - b2 ** 2 * np.sin(angle2) ** 2)
        
        a1 = np.cos(angle2) / np.sin(angle1)
        a2 = 1 / np.sin(angle1) * np.sqrt(-1 * np.cos(angle1) ** 2 + np.sin(angle2) ** 2 - b2 ** 2 * np.sin(angle2) ** 2)
        a3 = -1 * b2 * np.sin(angle2) / np.sin(angle1)
    elif gate == "Y":
        #b1 is a free parameter that must have magnitude at most sqrt(1 - cos^2(angle1)/sin^2(angle2))
        b1 = 0
        b2 = np.cos(angle1) / np.sin(angle2)
        b3 = 1 / np.sin(angle2) * np.sqrt(-1 * np.cos(angle1) ** 2 + np.sin(angle2) ** 2 - b1 ** 2 * np.sin(angle2) ** 2)
        
        a1 = -1 / np.sin(angle1) * np.sqrt(-1 * np.cos(angle1) ** 2 + np.sin(angle2) ** 2 - b1 ** 2 * np.sin(angle2) ** 2)
        a2 = np.cos(angle2) / np.sin(angle1)
        a3 = b1 * np.sin(angle2) / np.sin(angle1)
    else:
        raise Exception("unsupported gate " + gate)
    
    #synthesize the components into rotation axes alpha and beta
    beta = np.array([b1, b2, b3])
    alpha = np.array([a1, a2, a3])
    
    return alpha, beta, num_b_rotations

def identity_order(n, k):
    '''
    Computes the order in which the subspaces should be cleaned to the identity
    
    Inputs:
        n: the total number of subspaces in the system
        k: the 1-indexed number of the subspace that is to be X or Y (and not identity)
    
    Returns:
        two lists, the first of which contains the two indices of the subspaces to be cleaned first
        and the second of which contains the remaining indices of the subspaces to be cleaned next
    '''
    #account for all subspaces from 1 to n apart from k
    remaining_indices = list(range(1, k)) + list(range(k + 1, n + 1))
    two_pulse_spaces = []
    remaining_spaces = []
    
    #select two indices mu_1 and mu_2 to be cleaned according to the condition
    #sqrt(k/mu_1), sqrt(k/mu_2) are not integers
    for idx in range(len(remaining_indices)):
        subspace_idx = remaining_indices[idx]
        if k % subspace_idx != 0 or (isqrt(k // subspace_idx)) ** 2 != (k // subspace_idx):
            two_pulse_spaces.append(subspace_idx)
            if len(two_pulse_spaces) == 2:
                remaining_spaces.extend(remaining_indices[idx + 1:])
                break
        else:
            remaining_spaces.append(subspace_idx)
    return two_pulse_spaces, remaining_spaces

def perpendicular_vector(axis, theta = 0):
    '''
    Computes a vector perpendicular to a given axis with a degree of freedom
    
    Inputs:
        axis: the axis to which a perpendicular axis vector is found
        theta: a degree of freedom for the choice of perpendicular axis, ranges from 0 to 2*pi
    
    Returns:
        a vector perpendicular to the given vector
    '''
    #if the axis is the z-hat vector, take the x-hat vector as the first perpendicular vector
    if abs(axis[0]) < 1e-12 and abs(axis[1]) < 1e-12:
        perp_axis_1 = np.array([1, 0, 0])
    #otherwise, switch the x- and y-components and negate to obtain a perpendicular vector
    else:
        perp_axis_1 = np.array([-axis[1], axis[0], 0])
        perp_axis_1 = perp_axis_1 / np.linalg.norm(perp_axis_1)
    
    #use the cross product to find another perpendicular vector
    perp_axis_2 = np.cross(axis, perp_axis_1)
    perp_axis_2 = perp_axis_2 / np.linalg.norm(perp_axis_2)
    
    #find a perpendicular vector from the combination of these two perpendicular vectors
    #taking theta as a degree of freedom
    return np.cos(theta) * perp_axis_1 + np.sin(theta) * perp_axis_2

def pre_and_post_conjugation(rotation_sequence, subspace_num):
    '''
    Produces the subspace-independent axis-angle representations of the
    daggered and undaggered operations corresponding to a sequence on a given subspace
    
    Inputs:
        rotation_sequence: a list of 2-tuples specifying a sequence of axis-angle pulses
        subspace_num: the 1-indexed number of the subspace on which these pulses are to be applied
    
    Returns:
        two lists of 2-tuples, the first of which corresponds to the daggered subspace-independent version of the input sequence
        and the second of which corresponds to the undaggered subspace-independent version of the input sequence
    '''
    
    #reverse the sequence and negate the angles to produce the daggered version
    #divide angles by the square root of the subspace number such that the rotations are subspace-independent
    pre_conjugate_seq = [(axis, -angle / np.sqrt(subspace_num)) for (axis, angle) in reversed(rotation_sequence)]
    post_conjugate_seq = [(axis, angle / np.sqrt(subspace_num)) for (axis, angle) in rotation_sequence]
    
    return pre_conjugate_seq, post_conjugate_seq

def switch_axes_conjugation(init_axis, final_axis, subspace_num):
    '''
    Finds conjugation that switches a rotation about the initial axis to one about the final axis
    on the subspace given by subspace_num
    
    Inputs:
        init_axis (array): initial axis of the rotation on the subspace
        final_axis (array): final axis of the rotation on the subspace
        subspace_num (int): 1-indexed number of the subspace
    
    Returns:
        two lists of 2-tuples, the first of which corresponds to the part of the conjugation before the rotation
        and the second of which corresponds to the oart of the conjugation after the rotation
    '''
    axis_rotate, angle_rotate = axis_switch(init_axis, final_axis)
    rotation_sequence = xy_decomposition(axis_rotate, angle_rotate)
    pre_conjugate_seq, post_conjugate_seq = pre_and_post_conjugation(rotation_sequence, subspace_num)
    return pre_conjugate_seq, post_conjugate_seq


def produce_sequence(n, k, gate = "X", perp_thetas = 0, decomp_thetas = 0):
    '''
    Produces a sequence that places a particular non-identity gate on a given subspace
    and identity gates on all other subspaces for a finite-dimensional QO-Qudit
    
    Inputs:
        n (int): the number of subspaces in the finite quantum oscillator
        k (int): the number of the subspace on which to place a non-identity gate
        gate (str): the type of non-identity gate ("X" or "Y")
        perp_thetas (float/int or list): degrees of freedom for finding perpendicular axes for cleaning n - 3 subspaces
        decomp_thetas (float/int or list): degrees of freedom for xy-decomposition axes for cleaning n - 3 subspaces
        
    Returns:
        a list of 2-tuples with the axis-angle representations of the rotation sequence
        to be performed to attain the desired gate on subspace k of n subspaces
    '''
    
    #if the non-identity subspace number exceeds the number of subspaces, raise an error
    if k > n: raise ValueError("nonidentity gate position must be in range")
    #if either subspace number or count is not an integer, raise an error
    if type(k) is not int: raise TypeError("nonidentity gate position must be integer")
    if type(n) is not int: raise TypeError("number of subspaces must be integer")
    
    if n > 2:
        #processing degrees of freedom for choosing perpendicular and xy-decomposition axes
        if type(perp_thetas) is not list:
            perp_thetas = [perp_thetas for _ in range(n - 3)]
        if type(decomp_thetas) is not list:
            decomp_thetas = [decomp_thetas for _ in range(n - 3)]
        
        #raise exception if given degrees of freedom do not match those expected
        if len(perp_thetas) != (n - 3):
            raise ValueError("degrees of freedom for choice of perpendicular axes must be number of subspaces minus 3")
        if len(decomp_thetas) != (n - 3):
            raise ValueError("degrees of freedom for choice of xy-decomposition axes must be number of subspaces minus 3")
    
    #handle the case of one subspace with one direct rotation
    if n == 1:
        if gate == "X":
            return [(np.array([1, 0, 0]), np.pi)]
        elif gate == "Y":
            return [(np.array([0, 1, 0]), np.pi)]
        else:
            raise Exception("unsupported gate " + gate)
    #handle the other base case of two subspaces
    elif n == 2:
        phi1 = np.arccos(1 / np.tan(np.pi / np.sqrt(2)))
        phi2 = np.arccos(1 / np.tan(np.pi * np.sqrt(2)))
        if k == 1:
            seq = [(np.pi * np.sqrt(2), phi1), (np.pi / 2, 0), (np.pi * np.sqrt(2), phi1), (-np.pi / 2, 0)]
            if gate == "X":
                return [(phi_to_axis(phi), theta) for (theta, phi) in seq]
            elif gate == "Y":
                return [(phi_to_axis(phi + np.pi / 2), theta) for (theta, phi) in seq]
            else:
                raise Exception("unsupported gate " + gate)
        elif k == 2:
            seq = [(np.pi * 2, phi2), (np.pi / (2 * np.sqrt(2)), 0), (np.pi * 2, phi2), (-np.pi / (2 * np.sqrt(2)), 0)]
            if gate == "X":
                return [(phi_to_axis(phi), theta) for (theta, phi) in seq]
            elif gate == "Y":
                return [(phi_to_axis(phi + np.pi / 2), theta) for (theta, phi) in seq]
            else:
                raise Exception("unsupported gate " + gate)
    
    #initialize sequences
    pulse_sequence = []
    two_pulse_spaces, remaining_spaces = identity_order(n, k)
    
    n1 = two_pulse_spaces[0]
    n2 = two_pulse_spaces[1]
    phi1 = np.pi / np.sqrt(n1)
    phi2 = 2 * np.pi / np.sqrt(n2)
    #degree of freedom for 4-pulse sequence
    angle4 = 0
    #definition of 4-pulse sequence
    identity_pulse_sequence = [(np.array([-np.sin(angle4), np.cos(angle4), 0]), phi2),
                               (np.array([np.cos(angle4), np.sin(angle4), 0]), phi1),
                               (np.array([-np.sin(angle4), np.cos(angle4), 0]), phi2),
                               (np.array([np.cos(angle4), np.sin(angle4), 0]), -phi1)]
    
    #iterate over the remaining spaces to clean to identity
    for idx, n_idx in enumerate(remaining_spaces):
        #find the rotations on this subspace
        idx_rotations = [quaternion.from_rotation_vector(vec * angle * np.sqrt(n_idx)) for (vec, angle) in identity_pulse_sequence]
        
        #perform these rotations starting from the identity to find the final composite rotation on the subspace
        idx_subspace = np.quaternion(1, 0, 0, 0)
        for idx_rotation in idx_rotations:
            idx_subspace = idx_rotation * idx_subspace

        #find the angle and axis corresponding to the composite rotation on this subspace
        idx_subspace_vec = quaternion.as_rotation_vector(idx_subspace)
        idx_subspace_angle = np.linalg.norm(idx_subspace_vec)
        if abs(idx_subspace_angle) < 1e-10:
            continue
        idx_subspace_axis = idx_subspace_vec / idx_subspace_angle
        
        perp_axis = perpendicular_vector(idx_subspace_axis, perp_thetas[idx])
        identity_rotation_sequence = xy_decomposition(perp_axis, np.pi, decomp_thetas[idx])
        identity_pre_conjugate_seq, identity_post_conjugate_seq = pre_and_post_conjugation(identity_rotation_sequence, n_idx)
        
        identity_pulse_sequence = identity_pulse_sequence + identity_pre_conjugate_seq + identity_pulse_sequence + identity_post_conjugate_seq
    
    #apply the identity pulse sequence above on the subspace with the non-identity gate
    gate_subspace = np.quaternion(1, 0, 0, 0)
    gate_pulse_sequence = [quaternion.from_rotation_vector(vec * angle * np.sqrt(k)) for (vec, angle) in identity_pulse_sequence]
    for pulse in gate_pulse_sequence:
        gate_subspace = pulse * gate_subspace
    
    #extract the axis and angle of the rotation on the subspace with the non-identity gate
    gate_subspace_vec = quaternion.as_rotation_vector(gate_subspace)
    gate_subspace_angle = np.linalg.norm(gate_subspace_vec)
    gate_subspace_axis = gate_subspace_vec / gate_subspace_angle
    print("garbage", gate_subspace_vec)
    
    #determine the desired rotations with the angle of the rotation on the subspace with the non-identity gate
    #that produce the desired gate on this subspace
    alpha, beta, num_b_rotations = alpha_beta_decomposition(gate_subspace_angle, gate)
    
    #determine the sequences by which the rotation on the subspace
    #with the non-identity gate must be conjugated to yield the beta and alpha rotations
    beta_pre_conjugate_seq, beta_post_conjugate_seq = switch_axes_conjugation(gate_subspace_axis, beta, k)
    alpha_pre_conjugate_seq, alpha_post_conjugate_seq = switch_axes_conjugation(gate_subspace_axis, alpha, k)
    
    #repeat the rotation on the subspace with the non-identity gate and conjugate to produce the beta rotations
    pulse_sequence.extend(beta_pre_conjugate_seq)
    for _ in range(num_b_rotations):
        pulse_sequence.extend(identity_pulse_sequence)
    pulse_sequence.extend(beta_post_conjugate_seq)
    
    #conjugate the rotation on the subspace with the non-identity gate to produce the alpha rotation
    pulse_sequence.extend(alpha_pre_conjugate_seq)
    pulse_sequence.extend(identity_pulse_sequence)
    pulse_sequence.extend(alpha_post_conjugate_seq)
    
    #return the final pulse sequence
    return pulse_sequence

def find_final_quaternions(pulse_sequence, n):
    '''
    Finds the final quaternion representation of the rotation on each of the first n subspaces
    after the sequence of pulses given by pulse_sequence
    
    Inputs:
        pulse_sequence (list): a list of angle-axis tuples specifying the pulse sequence to be performed
    
    Returns:
        list of quaternions representing the composite rotations on the first n subspaces
    '''
    pulse_subspace_list = [[quaternion.from_rotation_vector(vec * angle * np.sqrt(s + 1)) for (vec, angle) in pulse_sequence] for s in range(n)]

    rotation_list = [np.quaternion(1, 0, 0, 0) for s in range(n)]

    for pulse_idx in range(len(pulse_sequence)):
        rotation_list = [pulse_subspace_list[s][pulse_idx] * rotation_list[s] for s in range(n)]
    
    return rotation_list

def find_final_angle_axis(pulse_sequence, n):
    '''
    Finds the final angle-axis representation of the rotation on each of the first n subspaces
    after the sequence of pulses given by pulse_sequence
    
    Inputs:
        pulse_sequence (list): a list of angle-axis tuples specifying the pulse sequence to be performed
    
    Returns:
        list of axis-angle tuples representing the composite rotations on the first n subspaces
    '''
    final_quaternions = find_final_quaternions(pulse_sequence, n)
    
    final_rotations = [quaternion.as_rotation_vector(q) for q in final_quaternions]
    
    return [(np.linalg.norm(vec), vec / np.linalg.norm(vec)) if np.linalg.norm(vec) > 1e-12 else (0, np.array([0.0, 0.0, 0.0])) for vec in final_rotations]

def find_theta_phi_representation(pulse_sequence):
    '''
    Finds theta, phi representation of sequence of axis-angle rotations,
    *assuming that all of the axes lie in the xy-plane*
    
    Inputs:
        pulse_sequence (list): a list of rotations in axis-angle tuple representation, must all be in xy-plane (i.e., have axis[2] = 0)
    
    Returns:
        list of tuples of theta-phi tuples representing the input rotation sequence
    '''
    return [(angle, np.arctan2(axis[1], axis[0])) for (axis, angle) in pulse_sequence]