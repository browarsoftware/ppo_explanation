using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using UnityEngine.Rendering;

public class CameraScript : MonoBehaviour
{
    // Start is called before the first frame update
    //RenderTexture rt = null;
    Camera Cam = null;
    //int FileCounter = 0;
    //string myPath = "c:/data/Camera/";
    public int width = 64;
    public int height = 64;
    public object lock_me = new object();
    void Start()
    {
        Cam = GetComponent<Camera>();
        m_Texture = new Texture2D(width, height, TextureFormat.RGB24, false);
        //rt = new RenderTexture(32, 32, 24);
        //rt = new RenderTexture(32, 32, 16, RenderTextureFormat.ARGB32);
    }

    int fileStrLen = 10;

    string getFileName(int value)
    {
        string myValue = value.ToString();
        while (myValue.Length < fileStrLen)
        {
            myValue = "0" + myValue;
        }
        return myValue;
    }
    public static void ObservationToTexture(Camera obsCamera, Texture2D texture2D, int width, int height)
    {
        if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null)
        {
            Debug.LogError("GraphicsDeviceType is Null. This will likely crash when trying to render.");
        }

        var oldRec = obsCamera.rect;
        obsCamera.rect = new Rect(0f, 0f, 1f, 1f);
        var depth = 24;
        var format = RenderTextureFormat.Default;
        var readWrite = RenderTextureReadWrite.Default;

        var tempRt =
            RenderTexture.GetTemporary(width, height, depth, format, readWrite);

        var prevActiveRt = RenderTexture.active;
        var prevCameraRt = obsCamera.targetTexture;

        // render to offscreen texture (readonly from CPU side)
        RenderTexture.active = tempRt;
        obsCamera.targetTexture = tempRt;

        obsCamera.Render();

        texture2D.ReadPixels(new Rect(0, 0, texture2D.width, texture2D.height), 0, 0);

        obsCamera.targetTexture = prevCameraRt;
        obsCamera.rect = oldRec;
        RenderTexture.active = prevActiveRt;
        RenderTexture.ReleaseTemporary(tempRt);
    }

    private byte[] compressed_image = null;
    public void saveImage(string FileName)
    {
        lock(lock_me)
        {
            if (compressed_image != null)
            {
                //File.WriteAllBytes(myPath + "/" + getFileName(FileCounter) + ".png", compressed_image);
                //FileCounter++;
                File.WriteAllBytes(FileName, compressed_image);
            }
        }
    }

    public Texture2D getTexture()
    {
        return m_Texture;
    }

    Texture2D m_Texture;
    void CamCapture()
    {
        Camera m_Camera = GetComponent<Camera>();
        //int m_Width = 64;
        //int m_Height = 64;
        // TODO support more types here, e.g. JPG
        lock (lock_me)
        {
            ObservationToTexture(m_Camera, m_Texture, width, height);
            //compressed_image = m_Texture.EncodeToPNG();
        }
        /*
        //Camera Cam = GetComponent<Camera>();

        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = Cam.targetTexture;

        Cam.Render();

        Texture2D Image = new Texture2D(Cam.targetTexture.width, Cam.targetTexture.height, TextureFormat.RGB24, false, false);
        Image.ReadPixels(new Rect(0, 0, Cam.targetTexture.width, Cam.targetTexture.height), 0, 0);
        Image.Apply();
        RenderTexture.active = currentRT;
        var Bytes = Image.EncodeToPNG();
        Destroy(Image);

        File.WriteAllBytes(myPath + "/" + getFileName(FileCounter) + ".png", Bytes);
        FileCounter++;*/
    }

    // Update is called once per frame
    void Update()
    {
        CamCapture();
    }
}
