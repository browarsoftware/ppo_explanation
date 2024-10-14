using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.Analytics;
using UnityEngine.PlayerLoop;

// https://www.youtube.com/watch?v=liWdLrv8pY0&list=PLhOLzjLZmaVdpnux85hqEGEBeqzcQsYNt&index=3
public class AgentController : Agent
{
    [SerializeField] private Transform target;
    [SerializeField] private float moveSpeed = 2.0f;//dla 10 się trenuje

    private Rigidbody rb;
    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }

    private void setRandomPosition()
    {
        target.transform.localPosition = new Vector3(8f * Random.value -4f,0, 8f * Random.value -4f);
        transform.localPosition = new Vector3(8f * Random.value -4f,0, 8f * Random.value -4f);
    }
    public override void OnEpisodeBegin()
    {
        //steps = 0;
        setRandomPosition();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        //sensor.AddObservation(target.localPosition);
    }
    //private int maxSteps = 1000;
    //private int steps = 0;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        float forward = actionBuffers.ContinuousActions[0];
        float rotate = actionBuffers.ContinuousActions[1];
        
        rb.MovePosition(transform.position + transform.forward * forward * Time.deltaTime * moveSpeed);
        transform.Rotate(0, rotate * moveSpeed, 0, Space.Self);

        /*
        Vector3 velocity = new Vector3(x, 0, z).normalized * moveSpeed * Time.deltaTime;
        transform.localPosition += velocity;
        steps++;
        AddReward(-0.01f);
        if (steps >= maxSteps)
        {
            AddReward(-1f);
            EndEpisode();
        }
        //AddReward(-0.001f);
        */
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    //public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> ca = actionsOut.ContinuousActions;
        ca[0] = -Input.GetAxisRaw("Vertical");
        ca[1] = Input.GetAxisRaw("Horizontal");
    }

    /*
    private override void OnCollisionEnter()
    {

    }*/
    
    private void OnTriggerEnter(Collider other)
    {
        Debug.Log(other.gameObject.tag);
        if (other.gameObject.tag == "target")
        {
            AddReward(4f);
            EndEpisode();
        }
        if (other.gameObject.tag == "wall")
        {
            AddReward(-1f);
            EndEpisode();
        }
    }
    /*
    void OnCollisionEnter(Collision col)
    {
        Debug.Log(col.transform.tag);
        if (col.transform.CompareTag("target"))
        {
            AddReward(10f);
            EndEpisode();
        }
        if (col.transform.CompareTag("wall"))
        {
            AddReward(-5f);
            EndEpisode();
        }
    }*/
}
