// Źródło https://www.immersivelimit.com/tutorials/simple-character-controller-for-unity
// https://www.immersivelimit.com/tutorials/ml-agents-camera-vision-coin-collector
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class InputController2 : MonoBehaviour
{
    private SimpleCharacterController2 charController;

    void Awake()
    {
        charController = GetComponent<SimpleCharacterController2>();
    }

    private void FixedUpdate()
    {
        // Get input values
        int vertical = Mathf.RoundToInt(Input.GetAxisRaw("Vertical"));
        int horizontal = Mathf.RoundToInt(Input.GetAxisRaw("Horizontal"));
        bool jump = Input.GetKey(KeyCode.Space);
        charController.ForwardInput = vertical;
        charController.TurnInput = horizontal;
        charController.JumpInput = jump;
    }
}
