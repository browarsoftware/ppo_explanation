// Opracowana na podstawie:
// https://www.immersivelimit.com/tutorials/simple-character-controller-for-unity
// https://www.immersivelimit.com/tutorials/ml-agents-camera-vision-coin-collector
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleCharacterController2 : MonoBehaviour
{
    [Tooltip("Maksymalne nachylenie terenu, powyżej ktorego nie uznajemy, że postać stoi na ziemi (nie może wtedy skakać)")]
    [Range(5f, 60f)]
    public float slopeLimit = 45f;
    [Tooltip("Szybkość ruchu (jednostki / sekundę)")]
    public float moveSpeed = 5f;
    [Tooltip("Szybkość obrotu w stopniach/sekundę, lewo - dodatni (+) prawo - ujemny (-)")]     
    public float turnSpeed = 300;
    [Tooltip("Czy obiekt może skakać")]
    public bool allowJump = false;
    [Tooltip("Szybkość ruchu w górę podczas skoku jednostka/sekundę")]
    public float jumpSpeed = 6f;
    public bool IsGrounded { get; private set; }
    public float ForwardInput { get; set; }
    public float TurnInput { get; set; }
    public bool JumpInput { get; set; }
    // obiekt, do którego przypinamy kontroler musi mieć dodaną bryłę sztywną i collider
    new private Rigidbody rigidbody;
    private CapsuleCollider capsuleCollider;
    // Funkcja Awake jest wywoływana podczas inicjowania obiektu skryptu, niezależnie od tego, czy skrypt jest włączony, czy nie. 
    // Start nie zostanie wywołana w tej samej chwili co Awake, jeśli skrypt nie jest włączony w czasie inicjalizacji.
    private void Awake()
    {
        // pobranie referencji do bryły sztywnej obiektu i jego collidera
        rigidbody = GetComponent<Rigidbody>();
        capsuleCollider = GetComponent<CapsuleCollider>();
    }
    /// <summary>
    /// Sprawdza, czy obiekt jest na ziemi i ustawia IsGrounded <see cref="IsGrounded"/>
    /// </summary>
    private void CheckGrounded()
    {
        IsGrounded = false;
        float capsuleHeight = Mathf.Max(capsuleCollider.radius * 2f, capsuleCollider.height);
        Vector3 capsuleBottom = transform.TransformPoint(capsuleCollider.center - Vector3.up * capsuleHeight / 2f);
        float radius = transform.TransformVector(capsuleCollider.radius, 0f, 0f).magnitude;
        Ray ray = new Ray(capsuleBottom + transform.up * .01f, -transform.up);
        RaycastHit hit;
        if (Physics.Raycast(ray, out hit, radius * 5f))
        {
            // bada kąt pomiędzy normalną do płaszczyzny, nad którą stoi obiekt a wektorem "w górę"
            float normalAngle = Vector3.Angle(hit.normal, transform.up);
            // jeśli kąt nachylenia podłoża jest mniejszy od maksymalnego dozwolonego
            if (normalAngle < slopeLimit)
            {
                // czy odległość pomiędzy dołem obiektu (kapsułki) a punktem zetknięcia promienia z podłożem jest mniejsza niż 0.02f
                // innymi słowy, heurystyka sprawdzająca, czy obiekt dotyka ziemi
                float maxDist = radius / Mathf.Cos(Mathf.Deg2Rad * normalAngle) - radius + .02f;
                if (hit.distance < maxDist)
                    IsGrounded = true;
            }
        }
    }
    /// <summary>
    /// Obliczanie ruchu obiektu
    /// </summary>
    private void ProcessActions()
    {
        // Obrót
        if (TurnInput != 0f)
        {
            float angle = Mathf.Clamp(TurnInput, -1f, 1f) * turnSpeed;
            transform.Rotate(Vector3.up, Time.fixedDeltaTime * angle);
        }
        // Przemieszczenie oraz skok dozwolone są jedynie na ziemi
        if (IsGrounded)
        {
            // wyzerowanie prędkości (dla uproszczenia nie ma bezwładności)
            rigidbody.velocity = Vector3.zero;
            // jeśli obiekt skacze, dodajemy prędkość skierowaną w górę
            if (JumpInput && allowJump)
            {
                rigidbody.velocity += Vector3.up * jumpSpeed;
            }

            // dodajemy wektor prędkości do w kierunku przód / tył
            rigidbody.velocity += transform.forward * Mathf.Clamp(ForwardInput, -1f, 1f) * moveSpeed;
        }
        else
        {
            // jeśli obiekt próbuje ruszać się przód/tył kiedy nie jest na ziemi
            if (!Mathf.Approximately(ForwardInput, 0f))
            {
                // Tworzymy nowy wektor prędkości, prędkość w płaszczyźnie góra/dół się nie zmienia
                // prędkość w płaszczyźnie przód-tył wynosi 0.5 możliwej szybkości
                Vector3 verticalVelocity = Vector3.Project(rigidbody.velocity, Vector3.up);
                rigidbody.velocity = verticalVelocity + transform.forward * Mathf.Clamp(ForwardInput, -1f, 1f) * moveSpeed / 2f;
            }
        }
    }
    // przy każdej kalkulacji dotyczącej interakcji z "fizycznym" światem
    private void FixedUpdate()
    {
        CheckGrounded();
        ProcessActions();
    }
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
