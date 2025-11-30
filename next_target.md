# Extentions:
**1.  Thumbs up. Logic will be simple: if only thumb is extended and thumb tip.y is less than pinky MCP then it's thumbs up**
**2.  Thumbs down. Logic will be simple: if only thumb is extended and thumb tip.y is greater than pinky MCP then it's thumbs down**
**3.  Thumbs up moving up. check if thumbs is up using the previous logic and if the velocity is more than a threshold in the up direction**
**4.  Thumbs up moving down. check if thumbs is up using the previous logic and if the velocity is more than a threshold in the down direction**
**5.  Thumbs down moving up. check if thumbs is down using the previous logic and if the velocity is more than a threshold in the up direction**
**6.  Thumbs down moving down. check if thumbs is down using the previous logic and if the velocity is more than a threshold in the down direction**

## important note: add any tunable metadata you will add here to the config file(eg. the threshold velocity and data like that are tunable) and perse it in the gesturemanager

**After you have done them create a .md file explaining what each filed in the config file mean and what tuning them does so it is easy for the user to tune them**
**Then create live change tracking system so the user can tune the configs on run:let me give you idea: create a gui app that shows the config file and allows the user to tune it in real time and see the changes in the app. when ever any value changes via the app it should trigger a callback that will update the gesturemanager and the gesturemanager should update the gesturemanager with the new config and also update the gui app with the new config**