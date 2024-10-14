// https://www.immersivelimit.com/tutorials/ml-agents-platformer-simple-coin-collector
// https://www.immersivelimit.com/tutorials/ml-agents-camera-vision-coin-collector
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using UnityEngine;

public class SimpleCollectorAgent2 : Agent
{
    [Tooltip("Obszar, po którym porusza się agent")]
    public GameObject platform;

    // pozycja startowa agenta
    private Vector3 startPosition;
    // obiekt obsługujący ruch agenta
    private SimpleCharacterController2 characterController;
    new private Rigidbody rigidbody;
    /// <summary>
    /// Wykonywane jeden raz podczas inicjalizacji agenta
    /// </summary>
    public override void Initialize()
    {
        startPosition = transform.position;
        // pobranie referencji do obiektu obsługującego ruch agenta oraz do bryły sztywnej
        characterController = GetComponent<SimpleCharacterController2>();
        rigidbody = GetComponent<Rigidbody>();
    }
    /// <summary>
    /// Wywołujemy za każdym razem, kiedy rozpoczyna się nowy epizod treningu agenta.
    /// </summary>
    public override void OnEpisodeBegin()
    {
        // ustawiamy agenta na pozycji startowej i zerujemy jego rotację oraz prędkość
        transform.position = startPosition;
        transform.rotation = Quaternion.Euler(Vector3.up * Random.Range(0f, 360f));
        rigidbody.velocity = Vector3.zero;

        // ustawiamy platformę na której leży cel w promieniu 5 jednostek od agenta
        Vector3 sp = startPosition 
            + Quaternion.Euler(Vector3.up * Random.Range(0f, 360f)) * Vector3.forward * 5f
            + new Vector3(0, -0.5f, 0);
        // korekcja wysokości
        platform.transform.position = new Vector3(sp.x, 1.7f, sp.z);
    }

    /// <summary>
    /// Controls the agent with human input
    /// </summary>
    /// <param name="actionsOut">The actions parsed from keyboard input</param>
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Read input values and round them. GetAxisRaw works better in this case
        // because of the DecisionRequester, which only gets new decisions periodically.
        int vertical = Mathf.RoundToInt(Input.GetAxisRaw("Vertical"));
        int horizontal = Mathf.RoundToInt(Input.GetAxisRaw("Horizontal"));
        bool jump = Input.GetKey(KeyCode.Space);

        // Convert the actions to Discrete choices (0, 1, 2)
        ActionSegment<int> actions = actionsOut.DiscreteActions;
        actions[0] = vertical >= 0 ? vertical : 2;
        actions[1] = horizontal >= 0 ? horizontal : 2;
        actions[2] = jump ? 1 : 0;
    }

    /// <summary>
    /// Zastosowanie akcji do agenta
    /// </summary>
    /// <param name="actions">The actions received</param>
    public override void OnActionReceived(ActionBuffers actions)
    {
        // Jeśli agent znajduje się zbyt daleko od pozycji początkowej zakończ epizod i dodaj karę
        if (Vector3.Distance(startPosition, transform.position) > 10f)
        {
            AddReward(-1f);
            EndEpisode();
        }

        // przekonwertuj akcje z dyskretnych wartości (0, 1, 2) na wartości (-1, 0, +1), które akceptuje kontroler akcji
        // wyjście o indeksie 0 to przód tył, o indeksie 1 to skręt, o indeksie 2 to skok
        float vertical = actions.DiscreteActions[0] <= 1 ? actions.DiscreteActions[0] : -1;
        float horizontal = actions.DiscreteActions[1] <= 1 ? actions.DiscreteActions[1] : -1;
        bool jump = actions.DiscreteActions[2] > 0;

        characterController.ForwardInput = vertical;
        characterController.TurnInput = horizontal;
        characterController.JumpInput = jump;
    }

    /// <summary>
    /// jeżeli agent zderzy się z innym colliderem
    /// </summary>
    /// <param name="other">The object (with trigger collider) that was touched</param>
    private void OnTriggerEnter(Collider other)
    {
        // jeżeli agent zderzył się z obiektem, który szukamy kończymy epizod i dostajemy nagrodę o wartości 1
        if (other.tag == "collectible")
        {
            AddReward(1f);
            EndEpisode();
        }
    }
}