using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.Analytics;

// Opracowane na podstawie
// https://www.youtube.com/watch?v=WgOGGnGdHeI
public class AgentScript : Agent
{
    // obiekt, w kierunku którego agent ma podążać, w naszym wypadku kulka
    [SerializeField] private Transform target;
    // szybkość ruchu agenta
    [SerializeField] private float moveSpeed = 2.0f;
    // maksymalna liczba ruchów wykonanych przez agenta, jeśli po ich wykonaniu nie dotrze do celu, próba kończy się porażką 
    [SerializeField] private int maxSteps = 1000;
    // liczba wykonanych ruchów w danej iteracji procesu uczenia agenta
    private int steps = 0;
    // inicjowanie losowej pozycji agenta i celu, zakładamy, że jest to losowy wektor [rand[-4, 4], 0, rand[-4, 4]]
    private void setRandomPosition()
    {
        target.transform.localPosition = new Vector3(8f * Random.value -4f,0, 8f * Random.value -4f);
        transform.localPosition = new Vector3(8f * Random.value -4f,0, 8f * Random.value -4f);
    }
    // metoda wykonywana na początku każdej iteracji (epizodu) procesu treningu agenta
    public override void OnEpisodeBegin()
    {
        steps = 0;
        setRandomPosition();
    }
    // pobieranie danych ze środowiska, obserwacje będą wykorzystywane w procesie uczenia agenta
    public override void CollectObservations(VectorSensor sensor)
    {
        // zapamiętujemy dwa wektory, w sumie 6 zmiennych typu float
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(target.localPosition);
    }
    // metoda wykonywana zarówno w procesie uczenia jak i podczas działania agenta - interpretuje odpowiedź modelu tłumacząc ją na ruch agenta
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // pobranie odpowiedzi z modelu, 2 zmienne typu float
        float x = actionBuffers.ContinuousActions[0];
        float z = actionBuffers.ContinuousActions[1];
        // policzenie przesunięcia i modyfikacja pozycji obiektu 
        Vector3 velocity = new Vector3(x, 0, z).normalized * moveSpeed * Time.deltaTime;
        transform.localPosition += velocity;
        steps++;
        // każdy ruch karany jest wartością -0.01f
        AddReward(-0.01f);
        // jeżeli nie udało się dotrzeć do celu po założonej liczbie kroków, kończymy iterację z dodatkową karą -1 
        if (steps >= maxSteps)
        {
            AddReward(-1f);
            EndEpisode();
        }
    }
    // zamiast modelu kierującego agentem możemy użyć poniżej heurystyki, która pobiera dane od użytkownika
    // aby działało, należy ustawić Behaviour Type na Heuristic only!
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> ca = actionsOut.ContinuousActions;
        ca[0] = Input.GetAxisRaw("Horizontal");
        ca[1] = Input.GetAxisRaw("Vertical");
    }

    // w wypadku kolizji z obiektem, jeżeli obiekt to target dostajemy nagrodę, 
    // jeśli to jest ścina, dostajemy karę
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
}
