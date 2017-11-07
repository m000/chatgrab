#!/bin/bash

# adb settings
ANDROID_SDK_HOME=${ANDROID_SDK_HOME:=./android-sdk}
ADB="$ANDROID_SDK_HOME"/platform-tools/adb
ADB_TARGET=localhost:5555
ADB_SWIPEUP="shell input swipe 600 100 600 260"

# quickgrab settings
QG="$HOME/Applications/QuickGrab/quickgrab"
QG_TITLE="droid"
QG_WID=$("$QG" -showlist yes | grep "$QG_TITLE" | awk -F'[:,]' '{print $6}' | tr -d \ )

# runtime settings
MAXITER=${MAXITER:=5000}
if [ "$SAVEAS" = "" ]; then
	if [ "$SAVEAS_PREFIX" != "" ]; then
		SAVEAS=$(printf "grab/%s-%%04d.png" "$SAVEAS_PREFIX")
	else
		SAVEAS=$(printf "grab/%%04d.png" "$SAVEAS_PREFIX")
	fi
fi

"$ADB" connect "$ADB_TARGET" | grep -q "connected to $ADB_TARGET"
if [ $? != 0 ]; then
	echo "Could not connect to adb."
	exit 1
fi

printf "Will run for %d iterations, or until stopped.\n" "$MAXITER"
printf "Sleeping for 5sec. Make sure that the VM window is clear!\n"
sleep 5

let i=0
while [ "$i" -lt "$MAXITER" ]; do
	fn=$(printf "$SAVEAS" "$i")
	printf "Grabbing %s...\n" "$fn"
	"$QG" -winid "$QG_WID" -file "$fn"
	sleep 3

	# three small swipe-ups are better than one big - avoids acceleration
	"$ADB" $ADB_SWIPEUP; sleep 1
	"$ADB" $ADB_SWIPEUP; sleep 1
	"$ADB" $ADB_SWIPEUP; sleep 2

	let i++
done

# "quickgrab -showlist yes" sample output:
#	App: VirtualBox VM, PID: 33373, Window ID: 14782, Window Title: droid (0) [Running]

