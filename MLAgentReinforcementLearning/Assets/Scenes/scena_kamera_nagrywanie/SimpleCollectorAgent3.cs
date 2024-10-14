// https://www.immersivelimit.com/tutorials/ml-agents-platformer-simple-coin-collector
// https://www.immersivelimit.com/tutorials/ml-agents-camera-vision-coin-collector
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using UnityEngine;

public class SimpleCollectorAgent3 : Agent
{
    [Tooltip("Obszar, po którym porusza się agent")]
    public GameObject platform;
    public Camera cam = null;
    //public Camera camSemantic = null;
    public RenderTexture camSemanticTexture = null;
    public RenderTexture overallRenderTexture = null;
    //public RenderTexture camRGBTexture = null;
    // pozycja startowa agenta
    private Vector3 startPosition;
    // obiekt obsługujący ruch agenta
    private CameraScript _CameraScript = null;
    //private CameraScript _CameraScriptSemantic = null;
    private SimpleCharacterController2 characterController;
    new private Rigidbody rigidbody;
    public string dirName = "c:/data/Camera/";
    private int fileStrLen = 6;
    private int fileId = 0;
    public bool saveToFile = true;
    public bool doNotMove = false;
    Texture2D texSemantic = null;
    Texture2D texOverall = null;
    public int texSemanticResolution = 192;
    public int texOverallResolution = 128;

    string getFileName(int value)
    {
        string myValue = value.ToString();
        while (myValue.Length < fileStrLen)
        {
            myValue = "0" + myValue;
        }
        return myValue;
    }
    /// <summary>
    /// Wykonywane jeden raz podczas inicjalizacji agenta
    /// </summary>
    public override void Initialize()
    {
        startPosition = transform.position;
        // pobranie referencji do obiektu obsługującego ruch agenta oraz do bryły sztywnej
        characterController = GetComponent<SimpleCharacterController2>();
        rigidbody = GetComponent<Rigidbody>();
        try
        {
            texSemantic = new Texture2D(texSemanticResolution, texSemanticResolution, TextureFormat.RGB24, false);
            texOverall = new Texture2D(texOverallResolution, texOverallResolution, TextureFormat.RGB24, false);

            _CameraScript = cam.GetComponent<CameraScript>();
            string _dateTimeHelper = System.DateTime.Now.ToString().Replace("\\", "_").Replace("/", "_").Replace(":","_");
            dirName += "/" + _dateTimeHelper + "/";
            bool exists = System.IO.Directory.Exists(dirName);
            if (!exists)
                System.IO.Directory.CreateDirectory(dirName);
        }
        catch { }
        /*try
        {
            _CameraScriptSemantic = camSemantic.GetComponent<CameraScript>();
        }
        catch { }*/
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

    Texture2D toTexture2D(RenderTexture rTex, Texture2D tex)
    {
        RenderTexture.active = rTex;
        tex.ReadPixels(new Rect(0, 0, rTex.width, rTex.height), 0, 0);
        tex.Apply();
        return tex;
    }


    /// <summary>
    /// Zastosowanie akcji do agenta
    /// </summary>
    /// <param name="actions">The actions received</param>
    public override void OnActionReceived(ActionBuffers actions)
    {
        ActionBuffers ab = GetStoredActionBuffers();
        var ca = ab.DiscreteActions.Array;
        string res = "";
        for (int a = 0; a < ca.Length; a++)
        {
            res += ca[a].ToString() + ",";
        }
        Debug.Log(res);
        try
        {
            var obs = GetObservations();
            Debug.Log(obs.Count);
            int a = 0;
            a++;
        }
        catch { }
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

        if (saveToFile)
        {
            if (cam != null && _CameraScript)
            {
                //    private int fileStrLen = 6;
                //private int fileId = 0;
                lock (_CameraScript.lock_me)
                {
                    Texture2D semanticT = null;
                    Texture2D agentCameraT = null;
                    Texture2D overallCameraT = null;
                    if (camSemanticTexture != null) semanticT = toTexture2D(camSemanticTexture, texSemantic);
                    if (overallRenderTexture != null) overallCameraT = toTexture2D(overallRenderTexture, texOverall);

                    string imageFileName = dirName + "/" + getFileName(fileId) + "_i.png";
                    string imageFileNameSemantic = dirName + "/" + getFileName(fileId) + "_s.png";
                    string overallFileNameSemantic = dirName + "/" + getFileName(fileId) + "_o.png";
                    //_CameraScript.saveImage(imageFileName);
                    agentCameraT = _CameraScript.getTexture();
                    if (agentCameraT != null)
                    {
                        System.IO.File.WriteAllBytes(imageFileName, agentCameraT.EncodeToPNG());
                    }
                    if (semanticT != null)
                    {
                        System.IO.File.WriteAllBytes(imageFileNameSemantic, semanticT.EncodeToPNG());
                    }
                    if (overallRenderTexture != null)
                    {
                        System.IO.File.WriteAllBytes(overallFileNameSemantic, overallCameraT.EncodeToPNG());
                    }

                    /*
                    if (camSemanticTexture != null)
                        System.IO.File.WriteAllBytes(dirName + "/" + getFileName(fileId) + "_Semantic.png",
                            toTexture2D(camSemanticTexture).EncodeToPNG());
                    */

                    using (System.IO.StreamWriter sw = System.IO.File.AppendText(dirName + "/results.txt"))
                    {
                        sw.Write(imageFileName + ",");
                        sw.Write(imageFileNameSemantic + ",");
                        sw.Write(overallFileNameSemantic + ",");
                        string actionsString = "";
                        for (int a = 0; a < actions.DiscreteActions.Length; a++)
                        {
                            if (a > 0)
                                actionsString += ",";
                            actionsString += actions.DiscreteActions[a];
                            //    sw.Write(",");
                            //sw.Write(actions.DiscreteActions[a]);
                        }
                        //sw.WriteLine("");
                        sw.WriteLine(actionsString);
                        Debug.Log(actionsString);
                    }
                    fileId++;
                }
            }
            /*
            if (cam != null && _CameraScriptSemantic)
            {
                //    private int fileStrLen = 6;
                //private int fileId = 0;
                lock (_CameraScriptSemantic.lock_me)
                {
                    string imageFileName = dirName + "/" + getFileName(fileId) + "_Semantic.png";
                    _CameraScriptSemantic.saveImage(imageFileName);
                    using (System.IO.StreamWriter sw = System.IO.File.AppendText(dirName + "/results.txt"))
                    {
                        sw.Write(imageFileName + ",");
                        string actionsString = "";
                        for (int a = 0; a < actions.DiscreteActions.Length; a++)
                        {
                            if (a > 0)
                                actionsString += ",";
                            actionsString += actions.DiscreteActions[a];
                            //    sw.Write(",");
                            //sw.Write(actions.DiscreteActions[a]);
                        }
                        //sw.WriteLine("");
                        sw.WriteLine(actionsString);
                        Debug.Log(actionsString);
                    }
                    //fileId++;
                }
            }*/
        }

        if (!doNotMove)
        {
            characterController.ForwardInput = vertical;
            characterController.TurnInput = horizontal;
            characterController.JumpInput = jump;
        }
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