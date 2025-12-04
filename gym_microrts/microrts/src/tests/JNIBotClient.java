package tests;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.file.Paths;

import ai.core.AI;
import ai.jni.Response;
import ai.reward.RewardFunctionInterface;
import gui.PhysicalGameStateJFrame;
import gui.PhysicalGameStatePanel;
import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.TraceEntry;
import rts.units.UnitTypeTable;

/**
 * Instances of this class each let us run a single environment (or sequence
 * of them, if we reset() in between) between two players. 
 * 
 * In this client, it is assumed that actions are selected by Java-based bots 
 * for **both** players. See JNIGridnetClient.java for a client where one
 * player is externally controlled, and JNIGridnetClientSelfPlay.java for one
 * where both players are externally controlled.
 *
 * @author santi and costa
 */
public class JNIBotClient {

    PhysicalGameStateJFrame w;
    public AI ai1;
    public AI ai2;
    PhysicalGameState pgs;
    GameState gs;
    UnitTypeTable utt;
    UnitTypeTable uttP0;
    UnitTypeTable uttP1;
    boolean partialObs;
    public RewardFunctionInterface[] rfs;
    public String mapPath;
    String micrortsPath;
    boolean gameover = false;
    boolean layerJSON = true;
    public int renderTheme = PhysicalGameStatePanel.COLORSCHEME_WHITE;

    // storage
    double[] rewards;
    boolean[] dones;
    Response response;
    PlayerAction pa1;
    PlayerAction pa2;

    /**
     * 
     * @param a_rfs Reward functions we want to use to compute rewards at every step.
     * @param a_micrortsPath Path for the microrts root directory (with Java code and maps).
     * @param a_mapPath Path (under microrts root dir) for map to load.
     * @param a_ai1
     * @param a_ai2
     * @param a_utt
     * @param partial_obs
     * @throws Exception
     */
    public JNIBotClient(RewardFunctionInterface[] a_rfs, String a_micrortsPath, String a_mapPath, AI a_ai1, AI a_ai2, UnitTypeTable a_utt, boolean partial_obs) throws Exception{
        micrortsPath = a_micrortsPath;
        mapPath = a_mapPath;
        rfs = a_rfs;
        utt = a_utt;
        uttP0 = a_utt;
        uttP1 = a_utt;
        partialObs = partial_obs;
        ai1 = a_ai1;
        ai2 = a_ai2;
        if (ai1 == null || ai2 == null) {
            throw new Exception("no ai1 or ai2 was chosen");
        }
        if (micrortsPath.length() != 0) {
            this.mapPath = Paths.get(micrortsPath, mapPath).toString();
        }

        pgs = PhysicalGameState.load(mapPath, uttP0);
        gs = new GameState(pgs, uttP0, uttP1);

        // initialize storage
        rewards = new double[rfs.length];
        dones = new boolean[rfs.length];
        response = new Response(null, null, null, null);
    }
    
    /**
     * Constructor for asymmetric UTT support
     * @param a_rfs
     * @param a_micrortsPath
     * @param a_mapPath
     * @param a_ai1
     * @param a_ai2
     * @param a_uttP0
     * @param a_uttP1
     * @param partial_obs
     * @throws Exception
     */
    public JNIBotClient(RewardFunctionInterface[] a_rfs, String a_micrortsPath, String a_mapPath, AI a_ai1, AI a_ai2, UnitTypeTable a_uttP0, UnitTypeTable a_uttP1, boolean partial_obs) throws Exception{
        micrortsPath = a_micrortsPath;
        mapPath = a_mapPath;
        rfs = a_rfs;
        utt = a_uttP0; // For backward compatibility
        uttP0 = a_uttP0;
        uttP1 = a_uttP1;
        partialObs = partial_obs;
        ai1 = a_ai1;
        ai2 = a_ai2;
        if (ai1 == null || ai2 == null) {
            throw new Exception("no ai1 or ai2 was chosen");
        }
        if (micrortsPath.length() != 0) {
            this.mapPath = Paths.get(micrortsPath, mapPath).toString();
        }

        pgs = PhysicalGameState.load(mapPath, uttP0);

        // initialize storage
        rewards = new double[rfs.length];
        dones = new boolean[rfs.length];
        response = new Response(null, null, null, null);
    }

    public byte[] render(boolean returnPixels) throws Exception {
        if (w==null) {
            w = PhysicalGameStatePanel.newVisualizer(gs, 640, 640, false, null, renderTheme);
        }
        w.setStateCloning(gs);
        w.repaint();

        if (!returnPixels) {
            return null;
        }
        BufferedImage image = new BufferedImage(w.getWidth(),
        w.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
        w.paint(image.getGraphics());

        WritableRaster raster = image .getRaster();
        DataBufferByte data = (DataBufferByte) raster.getDataBuffer();
        return data.getData();
    }

    public Response gameStep(int player) throws Exception {
        pa1 = ai1.getAction(player, gs);
        pa2 = ai2.getAction(1 - player, gs);

        gs.issueSafe(pa1);
        gs.issueSafe(pa2);
        TraceEntry te  = new TraceEntry(gs.getPhysicalGameState().clone(), gs.getTime());
        te.addPlayerAction(pa1.clone());
        te.addPlayerAction(pa2.clone());

        // simulate:
        gameover = gs.cycle();
        if (gameover) {
            ai1.gameOver(gs.winner());
            ai2.gameOver(gs.winner());
        }
        for (int i = 0; i < rewards.length; i++) {
            rfs[i].computeReward(player, 1 - player, te, gs);
            dones[i] = rfs[i].isDone();
            rewards[i] = rfs[i].getReward();
        }
        response.set(
            null,
            rewards,
            dones,
            "{}");
        return response;
    }

    public String sendUTT() throws Exception {
        Writer w = new StringWriter();
        utt.toJSON(w);
        return w.toString(); // now it works fine
    }

    /**
     * Return both players' UTTs as a composite JSON.
     * Now supports asymmetric UTTs.
     */
    public String sendUTTs() throws Exception {
        StringWriter w0 = new StringWriter();
        if (uttP0 != null) {
            uttP0.toJSON(w0);
        } else {
            utt.toJSON(w0);
        }
        String j0 = w0.toString();
        
        StringWriter w1 = new StringWriter();
        if (uttP1 != null) {
            uttP1.toJSON(w1);
        } else {
            utt.toJSON(w1);
        }
        String j1 = w1.toString();
        
        return "{\"p0\":" + j0 + ",\"p1\":" + j1 + "}";
    }
    
    /**
     * Get the current GameState. Useful for inspecting game state from Python.
     * @return the current GameState
     */
    public GameState getGameState() {
        return gs;
    }

    /**
     * Change UTT configuration at runtime during a game.
     * This method allows changing the Unit Type Table for one or both players
     * without restarting the game or modifying the Java environment.
     * 
     * @param newUttP0Json JSON string for Player 0's new UTT (null to keep current)
     * @param newUttP1Json JSON string for Player 1's new UTT (null to keep current)
     * @return true if successful, false otherwise
     * @throws Exception
     */
    public boolean changeUTT(String newUttP0Json, String newUttP1Json) throws Exception {
        try {
            UnitTypeTable newUttP0 = uttP0 != null ? uttP0 : utt;
            UnitTypeTable newUttP1 = uttP1 != null ? uttP1 : utt;
            
            // Parse new UTTs if provided
            if (newUttP0Json != null && !newUttP0Json.isEmpty()) {
                newUttP0 = UnitTypeTable.fromJSON(newUttP0Json);
            }
            if (newUttP1Json != null && !newUttP1Json.isEmpty()) {
                newUttP1 = UnitTypeTable.fromJSON(newUttP1Json);
            }
            
            // Clone game state with new UTTs
            GameState newGs = gs.cloneChangingUTTs(newUttP0, newUttP1);
            if (newGs == null) {
                return false; // cloneChangingUTTs failed (e.g., unit type mismatch)
            }
            
            // Update UTT references
            utt = newUttP0; // For backward compatibility
            uttP0 = newUttP0;
            uttP1 = newUttP1;
            
            // Update game state
            gs = newGs;
            
            // Update AI references with new UTTs
            // AIs that cache UTT need to be updated
            try {
                // Try to call reset(UnitTypeTable) method which most AIs have
                java.lang.reflect.Method resetMethod = ai1.getClass().getMethod("reset", UnitTypeTable.class);
                resetMethod.invoke(ai1, uttP0);
            } catch (Exception e) {
                // If reset(UnitTypeTable) doesn't exist, try JNIAI.setUTT
                if (ai1 instanceof ai.jni.JNIAI) {
                    ((ai.jni.JNIAI) ai1).setUTT(uttP0);
                }
            }
            
            try {
                java.lang.reflect.Method resetMethod = ai2.getClass().getMethod("reset", UnitTypeTable.class);
                resetMethod.invoke(ai2, uttP1);
            } catch (Exception e) {
                if (ai2 instanceof ai.jni.JNIAI) {
                    ((ai.jni.JNIAI) ai2).setUTT(uttP1);
                }
            }
            
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Resets the environment.
     * @param player This parameter is unused.
     * @return Response after reset.
     * @throws Exception
     */
    public Response reset(int player) throws Exception {
        ai1 = ai1.clone();
        ai1.reset();
        ai2 = ai2.clone();
        ai2.reset();
        pgs = PhysicalGameState.load(mapPath, uttP0);
        gs = new GameState(pgs, uttP0, uttP1);

        for (int i = 0; i < rewards.length; i++) {
            rewards[i] = 0;
            dones[i] = false;
        }
        response.set(
            null,
            rewards,
            dones,
            "{}");
        return response;
    }

    public void close() throws Exception {
        if (w!=null) {
            w.dispose();    
        }
    }
}
