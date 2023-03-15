#!/usr/bin/env python
# coding: utf-8

# # Titel des Projekts: Akkorderkennung24

# ## Beschreibung
# 
# Ziel des Projektes ist das Erstellen eines Python-Programms, welches einen Akkord (die 24 grundlegenden Dur-/Moll-Dreiklänge) aus einer Audiodatei erkennen und benennen kann.

# ## Eingangs- und Ausgangsdaten
# 
# **Eingangsdaten**<br>
# Im Notebook wird unter der Funktionsdeklaration eine Reihe von Demoaufrufen gestartet, welche spezifische wav-Audiodateien erwarten. Diese werden mit dem Notebook mitgeschickt. Ansonsten können bei eigener Benutzung der akkorderkennung()-Funktion aber auch alle anderen Audiodateien und Formate verwendet werden, die librosa üblicherweise verarbeiten kann(z.b. mp3). Es muss darauf geachtet werden, den korrekten Pfad zu übergeben (in Relation zu dem Ordner, in dem dieses Notebook hinterlegt ist.)
# Um die akkorderkennung()-Funktion sinnvollerweise nutzen zu können, sollte sich auf der übergebenen Audiodatei genau ein Akkord aus den 24 grundlegenden Dur-/Molldreiklängen befinden, andere Akkordtypen können nicht klassifiziert werden.
# 
# **Ausgangsdaten**<br>
# Jeder Aufruf der Funktion führt zu einer Audiowiedergabe, sowie einem textuellen Output.<br>
# Wird die Sinustonsektion ausgeführt, werden ebenso wav-Dateien mit Sinustonakkorden produziert und im selben Ordner wie dieses Notebook abgespeichert (diese werden zu Demozwecken weiter unten im Notebook analysiert). 

# ## Anleitung
# 
# Folgende nicht standardmäßig mit Anaconda mitgelieferten Libraries müssen installiert werden:<br>
#     - **librosa**<br>
#     - **sounddevice**<br>
# 
# Aufruf der Akkordklassifikation geschieht über die Funktion **akkorderkennung(audiopfad, description, percfilter):**<br>
#     - **audiopfad**: Pfad einer Audiodatei, die einen Akkord enthält (String)<br>
#     - **description**: Im Output kann ein Erklärtext mit ausgegeben werden, mithilfe dessen nachvollzogen werden kann, wie die             Funktion zu ihrem Ergebnis gelangt ist. Setze description=0, wenn kein Erklärtext gewünscht. Default ist description=1,         also wird standardmäßig der Erklärtext mit ausgegeben. (Integer)<br>
#     - **percfilter**: Es kann eine Bereinigung der Audiodatei vorgenommen werden, bei der perkussive Signale rausgefiltert werden.
#       Standardmäßig ist dies nicht der Fall, default ist percfilter=0. Falls Filterung gewünscht percfilter=1 setzen. (Integer)<br>
#       
# **Demoaufruf**: akkorderkennung('Audios/ADur.wav', description=1, percfilter=0) 

# ## Programmcode

# In[1]:


import librosa, librosa.display
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy
from matplotlib import cm
import numpy as np

def akkorderkennung(audiopfad, description=1, percfilter=0):
    sig1, sr1 = librosa.load(audiopfad, sr=44100)
    sd.play(sig1, sr1, blocking=True)

    ## Diagramm 1a
    # Darstellung der Chroma values der jeweiligen Audiodatei
    chroma_values = librosa.feature.chroma_stft(y=sig1, sr=sr1)
    plt.title('Chroma-Werte des Akkords')
    librosa.display.specshow(chroma_values, sr=sr1, x_axis='time', y_axis='chroma')
    ## onset zeug
    onset_envelope = librosa.onset.onset_strength(y=sig1, sr=sr1)
    size = onset_envelope.size 
    times = librosa.frames_to_time(np.arange(onset_envelope.size), sr=sr1)
    plt.plot(times, onset_envelope, label='Onset strength')
    plt.legend()
    #liste mit peaks in onset kurve
    onset_frames = librosa.onset.onset_detect(y=sig1, sr=sr1)
    onset_times = librosa.onset.onset_detect(y=sig1, sr=sr1, units='time') # unit: in sekunden anzeigen statt frames
    plt.vlines(onset_times, 0, onset_envelope.max(), color='g', linestyle='--', label='Onsets')
    plt.legend()
    plt.show()
  
    if percfilter == 1:
        ## Diagramm 1b
        # Darstellung der Chroma values der jeweiligen Audiodatei nach Herausfilterung perkussiver Anteile
        sig_harmonic, sig_percussive = librosa.effects.hpss(sig1)
        # spectogramm brauche ich nicht wirklich...
        #spectogram_data_harmonic = librosa.stft(sig_harmonic)
        #magnitudes_harmonic = librosa.amplitude_to_db(np.abs(spectogram_data_harmonic))
        #plt.figure()
        #plt.title('Spectogramm des Akkords nach Perkussionsfilterung')
        #librosa.display.specshow(magnitudes_harmonic, x_axis='time', y_axis='log')
        #plt.show()
        sd.play(sig_harmonic, sr1, blocking=True)
        sd.play(sig_percussive, sr1, blocking=True)
        plt.figure()
        plt.title('Chroma-Werte des Akkords nach Perkussionsfilterung')
        chroma_values_harmonic = librosa.feature.chroma_stft(y=sig_harmonic, sr=sr1)
        librosa.display.specshow(chroma_values_harmonic, sr=sr1, x_axis='time', y_axis='chroma')
        plt.show()
    
    
    
    ## Stille vor und nach Akkord abschneiden und Plot zum Vergleich zu vorher anzeigen
    #sig_trimmed, index = librosa.effects.trim(y=sig1, top_db=70)
    #chroma_values_trimmed = librosa.feature.chroma_stft(y=sig_trimmed, sr=sr1)
    #librosa.display.specshow(chroma_values_trimmed, sr=sr1, x_axis='time', y_axis='chroma')
    #plt.show()
    #print("Trimmed stuff:")
    #print(librosa.get_duration(y=sig_trimmed), librosa.get_duration(y=sig1), librosa.get_duration(y=index))
    # scheint nicht zu funktionieren, gibt mir immer die gleiche Länge vor und nach trimming an...
    
    ## Diagramm 2
    # wavediagramm, einfach weil nett es zu sehen
    librosa.display.waveshow(y=sig1, sr=sr1)
    plt.title('Akkord als Wellendiagramm')
    plt.show()


    ## Summen über einzelne Chroma bilden, um stärkste Frequenzen herauszubekommen
    # zuvor noch bestimmen, mit welchen Chroma values fortan gearbeitet werden soll- mit oder ohne Perkussionsfilterung?
    if percfilter == 1:
        chroma_values = chroma_values_harmonic
        
    # Einzellisten der chromas aus dem 2d chroma values array erstellen
    blist = chroma_values[11]
    aislist = chroma_values[10]
    alist = chroma_values[9]
    gislist = chroma_values[8]
    glist = chroma_values[7]
    fislist = chroma_values[6]
    flist = chroma_values[5]
    elist = chroma_values[4]
    dislist = chroma_values[3]
    dlist = chroma_values[2]
    cislist = chroma_values[1]
    clist = chroma_values[0]

    # die Werte in jeder Liste aufsummieren
    sum_blist = sum(blist)
    sum_aislist = sum(aislist)
    sum_alist = sum(alist)
    sum_gislist = sum(gislist)
    sum_glist = sum(glist)
    sum_fislist = sum(fislist)
    sum_flist = sum(flist)
    sum_elist = sum(elist)
    sum_dislist = sum(dislist)
    sum_dlist = sum(dlist)
    sum_cislist = sum(cislist)
    sum_clist = sum(clist)

    #Dict anlegen, um an Werte zu kommen ohne bezug zu verlieren zur Tonbeschriftung
    Tonsum_dict ={'c': sum_clist, 'cis': sum_cislist, 'd': sum_dlist, 'dis': sum_dislist, 'e': sum_elist, 'f': sum_flist, 'fis': sum_fislist, 'g': sum_glist, 'gis': sum_gislist, 'a': sum_alist, 'ais': sum_aislist, 'b': sum_blist}
    sorted_Tonsum_dict = sorted(Tonsum_dict.items(), key=lambda x: x[1], reverse=True)
    #print(sorted_Tonsum_dict)
            
   
    # Liste aus den 3 stärksten bilden und sie mit vorgefertigeten Akkordmusterlisten abgleichen

    ## Akkordmuster anlegen. Hier wird verzeichnet, welche Töne ein bestimmter Akkord enthält
    # Dur-Akkorde
    bdur = {11, 3, 6}
    aisdur_bbdur = {10, 2, 5}
    adur = {9, 1, 4}
    gisdur_asdur = {8, 0, 3}
    gdur = {7, 11, 2}
    fisdur_gesdur = {6, 10, 1}
    fdur = {5, 9, 0}
    edur = {4, 8, 11}
    disdur_esdur = {3, 7, 10}
    ddur = {2, 6, 9}
    cisdur_desdur = {1, 5, 8}
    cdur = {0, 4, 7}
    # Moll-Akkorde
    bm = {11, 2, 6}
    aism_bbm = {10, 1, 5}
    am = {9, 0, 4}
    gism_asm = {8, 11, 3}
    gm = {7, 10, 2}
    fism_gesm = {6, 9, 1}
    fm = {5, 8, 0}
    em = {4, 7, 11}
    dism_esm = {3, 6, 10}
    dm = {2, 5, 9}
    cism_desm = {1, 4, 8}
    cm = {0, 3, 7}
    
    
    ### es folgen einige Helperfunktionen, die später benutzt werden
    
    # mapping von nummern zu noten
    def chord_numbers_to_notes(chord):
        chord_notes = set()
        for item in chord:
            if item == 0:
                chord_notes.add('c')
            elif item == 1:
                chord_notes.add('cis/des')
            elif item == 2:
                chord_notes.add('d')
            elif item == 3:
                chord_notes.add('dis/es')
            elif item == 4:
                chord_notes.add('e')
            elif item == 5:
                chord_notes.add('f')
            elif item == 6:
                chord_notes.add('fis/ges')
            elif item == 7:
                chord_notes.add('g')
            elif item == 8:
                chord_notes.add('gis/as')
            elif item == 9:
                chord_notes.add('a')
            elif item == 10:
                chord_notes.add('ais/bb')
            elif item == 11:
                chord_notes.add('b')
            else:
                print('Eine Zahl ist im Akkord, die nicht zu einer Note gematcht werden kann.')
        return chord_notes
    
    def note_shift(note_int):
        if note_int >=0 and note_int <=10:
            note_int += 1
        elif note_int == 11:
            note_int = 0
        else:
            print('Eine Note die nicht existiert soll geshiftet werden - kleiner 0 oder höher 11')
            
        return note_int
    
    
    def note_is_neighbour(note1, note2):
        if (note1 + 1) == note2 or (note2 + 1) == note1:
            return True
        elif note1 == 11 and note2 == 0 or note1 == 0 and note2 == 11:
            return True
        else:
            return False
        
    def is_q(n1, n2):
        # überprüft, ob Noten n1 und n2 im Quintabstand zueinander liegen
        # mod in case addition of 7 crosses limit of 11
        if ((n1 + 7) % 12) == n2:
            return True
        else:
            return False
        
        
    def make_g3(prime):
        # große Terz von Prime bestimmen...4 mal nach oben shiften
        g3 = note_shift(prime)
        g3 = note_shift(g3)
        g3 = note_shift(g3)
        g3 = note_shift(g3)
        
        return g3
    
    
    def make_k3(prime):
        # kleine Terz von Grundton aus bestimmen...3 mal nach oben shiften
        k3 = note_shift(prime)
        k3 = note_shift(k3)
        k3 = note_shift(k3)
       
        return k3
    
       
    def if5_put3(prime, quinte, note3):
        # wird im Fall aufgerufen, dass genau 2 benachbarte Töne im detected chord drin stecken(note3 = irrelevanter Nachbar)
        # bestimmt ob Quinte im detected chord enthalten ist und steckt dann passende terz dazu hinein
        
        # bestimmen, ob Quinte schon enthalten
        if is_q(prime, quinte):                 #### teilweise doppelte quintabfrage!nicht sicher, ob immmer doppelt? drin lassen.
            if description == 1:
                print('Quinte im Akkord gefunden...passendere Terz suchen')
            # den 3. Ton, der nicht Teil der Quinte ist rauswerfen
            remove_this = note3
            # aus liste oder set rauswerfen? set oder?
            detected_chord.remove(remove_this)
            # bestimmen, welche Töne stattdessen in Frage kommen: kl u gr Terz
            kl_terz = make_k3(prime)
            if description == 1:
                print(f'die kleine Terz ist {kl_terz}')
            gr_terz = make_g3(prime)
            if description == 1:
                print(f'die große Terz ist {gr_terz}')
            # überprüfen, welcher davon die höheren Chroma values besitzt und diesen in den detected chord aufnehmen
            # TODO: ändern, falls chroma stärke nicht mehr einfach nur über chroma array summe gebildet wird
            strength_kl_terz = sum(chroma_values[kl_terz])
            strength_gr_terz = sum(chroma_values[gr_terz])
            if strength_kl_terz > strength_gr_terz:
                detected_chord.add(kl_terz)
                if description == 1:
                    print(f'kl_terz eingefügt: {detected_chord}')
            elif strength_kl_terz < strength_gr_terz:
                detected_chord.add(gr_terz)
                if description == 1:
                    print(f'gr_terz eingefügt: {detected_chord}')
            else:
                print('wierd')
        
        
        
    def localize_q_and_put3(detected_chord, round=1):
        # diese Funktion prüft, ob sich eine Quinte innerhalb der Akkordliste befindet und steckt passende Terz in den detected_chord
        # Da die Position der Quinte unklar ist, müssen mehrere Positionen überprüft werden
        
        # erstmal sortieren, damit man nur bei Aufwärtsbewegung von den potentiellen Quintschritten ausgehen kann
        # und n icht auch noch rückwärts testen muss
        list_detected_chord = list(detected_chord)
        ordered_detected_chord_list = list_detected_chord.sort()
        
        n0 = list_detected_chord[0]
        n1 = list_detected_chord[1]
        n2 = list_detected_chord[2]
        
        if is_q(n0, n1):
            the_prime = n0
            the_quinte = n1
            the_no3 = n2
            if5_put3(the_prime, the_quinte, the_no3)
            
        elif is_q(n0, n2):
            the_prime = n0
            the_quinte = n2
            the_no3 = n1
            if5_put3(the_prime, the_quinte, the_no3)
            
        elif is_q(n1, n2):
            the_prime = n1
            the_quinte = n2
            the_no3 = n0
            if5_put3(the_prime, the_quinte, the_no3)
            
        elif is_q(n1, n0):
            the_prime = n1
            the_quinte = n0
            the_no3 = n2
            if5_put3(the_prime, the_quinte, the_no3)
            
        elif is_q(n2, n0):
            the_prime = n2
            the_quinte = n0
            the_no3 = n1
            if5_put3(the_prime, the_quinte, the_no3)
            
        elif is_q(n2, n1):
            the_prime = n2
            the_quinte = n1
            the_no3 = n0
            if5_put3(the_prime, the_quinte, the_no3)
        else:
            if round == 1:
                if description == 1:
                    print('Keine Quinte trotz benachbarter Töne gefunden')
            if round == 2:
                if description == 1:
                    print('Keine Quinte innerhalb des Akkordsets vorhanden...dementsprechend kann auch keine passende Terz ausfindig gemacht werden.')
            
         
                

    ### Ende der Helperfunktionen- nun geht es mit dem Hauptprogramm weiter...

    
    # Liste aus allen Akkorden machen um später abfragen zu können ob einer dazu zu dem detected chord passt
    # erstmal ein set aus den 3 stärksten Tönen zum Abgleichen bilden
    
    # Initialisierung potenzieller benachbarter Töne + leeres set um töne in fkt einzufüllen
    avoid_b = 999
    avoid_a = 999
    detected_chord_zero = set()
    
    def three_strongest(avoid_before, avoid_after, detected_chord):
        # diese Funktion bestimmt die 3 Töne, die die höchsten Chroma values besitzen und fügt sie in ein set ein
        if description == 1:
            print('\nDie 3 stärksten Töne des Akkords werden gesucht...(Funktion three_strongest() arbeitet)')
        counter = 0
        for key, value in sorted_Tonsum_dict:
            counter += 1
            if counter <= 3:
                if description == 1:
                    print(f'Der counter steht auf {counter}')
                
                if key == 'b':
                    if description == 1:
                        print('11 wird in Betracht gezogen für den Akkord')
                    if len(detected_chord) == 3:
                        if avoid_before == 11:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        elif avoid_after == 11:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        else:
                            detected_chord.add(11)
                    else:
                        detected_chord.add(11)
                    
                elif key == 'ais':
                    if description == 1:
                        print('10 wird in Betracht gezogen für den Akkord.')
                    if len(detected_chord) == 3:
                        if avoid_before == 10:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter = counter - 1
                        elif avoid_after == 10:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter = counter - 1
                        else:
                            detected_chord.add(10)
                    else:
                        detected_chord.add(10)
                    
                elif key == 'a':
                    if description == 1:
                        print('9 wird in Betracht gezogen für den Akkord.')
                    if len(detected_chord) == 3:
                        if avoid_before == 9:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        elif avoid_after == 9:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        else:
                            detected_chord.add(9)
                    else:
                        detected_chord.add(9)
                    
                elif key == 'gis':
                    if description == 1:
                        print('8 wird in Betracht gezogen für den Akkord.')
                    if len(detected_chord) == 3:
                        if avoid_before == 8:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        elif avoid_after == 8:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        else:
                            detected_chord.add(8)
                    else:
                        detected_chord.add(8)
                    
                elif key == 'g':
                    if description == 1:
                        print('7 wird in Betracht gezogen für den Akkord')
                    if len(detected_chord) == 3:
                        if avoid_before == 7:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        elif avoid_after == 7:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        else:
                            detected_chord.add(7)
                    else:
                        detected_chord.add(7)
                    
                elif key == 'fis':
                    if description == 1:
                        print('6 wird in Betracht gezogen für den Akkord')
                    if len(detected_chord) == 3:
                        if avoid_before == 6:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        elif avoid_after == 6:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        else:
                            detected_chord.add(6)
                    else:
                        detected_chord.add(6)
                    
                elif key == 'f':
                    if description == 1:
                        print('5 wird in Betracht gezogen für den Akkord.')
                    if len(detected_chord) == 3:
                        if avoid_before == 5:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        elif avoid_after == 5:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        else:
                            detected_chord.add(5)
                    else:
                        detected_chord.add(5)
                    
                elif key == 'e':
                    if description == 1:
                        print('4 wird in Betracht gezogen für den Akkord.')
                    if len(detected_chord) == 3:
                        if avoid_before == 4:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        elif avoid_after == 4:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        else:
                            detected_chord.add(4)
                    else:
                        detected_chord.add(4)
                    
                elif key == 'dis':
                    if description == 1:
                        print('3 wird in Betracht gezogen für den Akkord.')
                    if len(detected_chord) == 3:
                        if avoid_before == 3:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        elif avoid_after == 3:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        else:
                            detected_chord.add(3)
                    else:
                        detected_chord.add(3)
                    
                elif key == 'd':
                    if description == 1:
                        print('2 wird in Betracht gezogen für den Akkord.')
                    if len(detected_chord) == 3:
                        if avoid_before == 2:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        elif avoid_after == 2:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        else:
                            detected_chord.add(2)
                    else:
                        detected_chord.add(2)
                    
                elif key == 'cis':
                    if description == 1:
                        print('1 wird in Betracht gezogen für den Akkord.')
                    if len(detected_chord) == 3:
                        if avoid_before == 1:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        elif avoid_after == 1:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        else:
                            detected_chord.add(1)
                    else:
                        detected_chord.add(1)
                    
                elif key == 'c':
                    if description == 1:
                        print('0 wird in Betracht gezogen für den Akkord.')
                    if len(detected_chord) == 3:
                        if avoid_before == 0:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        elif avoid_after == 0:
                            if description == 1:
                                print('Muss vermieden werden.')
                            counter -= 1
                        else:
                            detected_chord.add(0)
                    else:
                        detected_chord.add(0)
                    
        
                else: print('Error- etwas konnte nicht gematcht werden für den Akkord.')
                
        # remove the avoid notes from set
        if avoid_b in detected_chord:
            detected_chord.remove(avoid_b)
        if avoid_a in detected_chord:
            detected_chord.remove(avoid_a)
        
        return detected_chord
        
        
    ## einmal die 3 stärksten töne herausfischen mit der funktion
    # beim ersten Aufruf von three_strongest werden einfach die 3 stärksten Töne in ein zuvor leeres Set eingefügt.
    # die avoid notes sind hier noch irrelevant, da sie 999 als dummy enthalten
    detected_chord = three_strongest(avoid_b, avoid_a, detected_chord_zero)
    if description == 1:
        print(f'Das ist der detected_chord nach Aufruf von three_strongest(): {detected_chord}')
                
    # den detected chord dahingehend untersuchen, ob benachbarte Töne aufgenommen wurden
    list_detected_chord = list(detected_chord)
    ordered_detected_chord_list = list_detected_chord.sort()
    if description == 1:
        print('Ausgabe einer sortierten Akkordliste:')
        print(list_detected_chord)
   

    ## 3er tonreihen bei Identifizierung der 3 stärksten Chromas abfangen und neu berechnen
    if description == 1:
        print('\n')
        print('Test des Akkordsets auf das Enthalten 3er benachbarter Töne...')
    if note_is_neighbour(list_detected_chord[0], list_detected_chord[1]) and note_is_neighbour(list_detected_chord[1], list_detected_chord[2]):
        if description == 1:
            print('3 aufeinanderfolgende Töne sind im detected_chord gelandet!')
        # diese Töne vermeiden
        avoid_b= list_detected_chord[0]
        avoid_a= list_detected_chord[2]
        # then do the detection for the strongest notes again, but avoid the adjacent notes
        detected_chord = three_strongest(avoid_b, avoid_a, detected_chord)
        if description == 1:
            print(f'Das ist der neue detected_chord nachdem die Nachbartöne gefiltert wurden: {detected_chord}')
    elif note_is_neighbour(list_detected_chord[2], list_detected_chord[0]) and note_is_neighbour(list_detected_chord[0], list_detected_chord[1]):
        if description == 1:
            print('3 aufeinanderfolgende Töne sind im detected_chord gelandet!')
        # diese Töne vermeiden
        avoid_b= list_detected_chord[2]
        avoid_a= list_detected_chord[1]
        # then do the detection for the strongest notes again, but avoid the adjacent notes
        detected_chord = three_strongest(avoid_b, avoid_a, detected_chord)
        if description == 1:
            print(f'Das ist der neue detected_chord nachdem die Nachbartöne gefiltert wurden: {detected_chord}')
    elif note_is_neighbour(list_detected_chord[2], list_detected_chord[0]) and note_is_neighbour(list_detected_chord[1], list_detected_chord[2]):
        if description == 1:
            print('3 aufeinanderfolgende Töne sind im detected_chord gelandet!')
        # diese Töne vermeiden
        avoid_b= list_detected_chord[1]
        avoid_a= list_detected_chord[0]
        # then do the detection for the strongest notes again, but avoid the adjacent notes
        detected_chord = three_strongest(avoid_b, avoid_a, detected_chord)
        if description == 1:
            print(f'Das ist der neue detected_chord nachdem die Nachbartöne gefiltert wurden: {detected_chord}')
    else:
        if description == 1:
            print('Keine 3 benachbachbarten Töne gefunden.')
    
    ## 2er tonreihen finden und die Quinte stärken falls vorhanden (falls Quinte vorhanden, weiß man welcher der benachbarten der störende Ton ist)   
    if description == 1:
        print('\n')
        print('Test des Akkordsets auf 2 benachbarte Töne...')
    # erstmal noch den neuen detected chord wieder sortieren
    list_detected_chord = list(detected_chord)
    ordered_detected_chord_list = list_detected_chord.sort()
    # nur nach Quinte suchen, wenn benachbarte Töne enthalten sind
    #...ansonsten später nochmal nach Quinte suchen, wenn match value nur 2
    if note_is_neighbour(list_detected_chord[0], list_detected_chord[1]) or note_is_neighbour(list_detected_chord[1], list_detected_chord[2]) or note_is_neighbour(list_detected_chord[0], list_detected_chord[2]):
        if description == 1:
            print('2 aufeinanderfolgende Töne sind im detected_chord gelandet!')
        localize_q_and_put3(detected_chord, round=1)
    else:
        if description == 1:
            print('Keine 2 benacharten Töne gefunden.')
                
    
    def classify_detected_chord(detected_chord):
        ## die Wahrscheinlichkeit bestimmen, dass der detected chord einem bestimmten Akkordpattern entspricht
    
        # erstmal leere Wahrscheinlichkeiten für alle Akkorde anlegen
        # Dur-Akkorde Wahrscheinlichkeit
        cdur_prob = 0
        cisdur_desdur_prob = 0
        ddur_prob = 0
        disdur_esdur_prob = 0
        edur_prob = 0
        fdur_prob = 0
        fisdur_gesdur_prob = 0
        gdur_prob = 0
        gisdur_asdur_prob = 0
        adur_prob = 0
        aisdur_bbdur_prob = 0
        bdur_prob = 0
        # Moll_akkorde Wahrscheinlichkeit     
        cm_prob = 0
        cism_desm_prob = 0
        dm_prob = 0
        dism_esm_prob = 0
        em_prob = 0
        fm_prob = 0
        fism_gesm_prob = 0
        gm_prob = 0
        gism_asm_prob = 0
        am_prob = 0
        aism_bbm_prob = 0
        bm_prob = 0
    
        # Jedem Akkord Wahrscheinlichkeitspunkte dafür geben, wenn die 3 stärksten Töne im jeweiligen Akkord vorkommen. 
        # Ziel: Akkordmuster mit höchsten Punkten herausfinden
        for item in detected_chord:
            # Dur
            if item in cdur:
                cdur_prob += 1
            if item in cisdur_desdur:
                cisdur_desdur_prob += 1
            if item in ddur:
                ddur_prob += 1
            if item in disdur_esdur:
                disdur_esdur_prob += 1
            if item in edur:
                edur_prob += 1
            if item in fdur:
                fdur_prob += 1
            if item in fisdur_gesdur:
                fisdur_gesdur_prob += 1
            if item in gdur:
                gdur_prob += 1
            if item in gisdur_asdur:
                gisdur_asdur_prob += 1
            if item in adur:
                adur_prob += 1
            if item in aisdur_bbdur:
                aisdur_bbdur_prob += 1
            if item in bdur:
                bdur_prob += 1
            # Moll
            if item in cm:
                cm_prob += 1
            if item in cism_desm:
                cism_desm_prob += 1
            if item in dm:
                dm_prob += 1
            if item in dism_esm:
                dism_esm_prob += 1
            if item in em:
                em_prob += 1
            if item in fm:
                fm_prob += 1
            if item in fism_gesm:
                fism_gesm_prob += 1
            if item in gm:
                gm_prob += 1
            if item in gism_asm:
                gism_asm_prob += 1
            if item in am:
                am_prob += 1
            if item in aism_bbm:
                aism_bbm_prob += 1
            if item in bm:
                bm_prob += 1
        
        
        # dict der Akkordwahrscheinlichkeiten anlegen
        prob_dict = {'cdur': cdur_prob, 'cisdur_desdur': cisdur_desdur_prob, 'ddur': ddur_prob, 'disdur_esdur': disdur_esdur_prob, 'edur': edur_prob, 'fdur': fdur_prob, 'fisdur_gesdur': fisdur_gesdur_prob, 'gdur': gdur_prob, 'gisdur_asdur': gisdur_asdur_prob, 'adur': adur_prob, 'aisdur_bbdur': aisdur_bbdur_prob, 'bdur': bdur_prob, 'cm': cm_prob, 'cism_desm': cism_desm_prob, 'dm': dm_prob, 'dism_esm': dism_esm_prob, 'em': em_prob, 'fm': fm_prob, 'fism_gesm': fism_gesm_prob, 'gm': gm_prob, 'gism_asm': gism_asm_prob, 'am': am_prob, 'aism_bbm': aism_bbm_prob, 'bm': bm_prob}
        #print(prob_dict)

   
    
        # Initialisierungen
        # Abspeichern, wie viele Töne bei dem Matching tatsächlich überein gestimmt haben. Match-value variable initialisieren
        match_value = 0
        # Initialisierung der Akkordklassifikation
        chord_classification = 'no chord classification available yet'

        #abfragen, ob 3er match vorliegt
        for key in prob_dict:
            if prob_dict[key] == 3:
                chord_classification = key
                match_value = 3
        # Abfragen, ob 2er match vorliegt, wenn kein 3er match vorhanden ist 
        # set anlegen, um alle 2er zu speichern
        twoset = set()
        if chord_classification == 'no chord classification available yet':
            for key in prob_dict:
                if prob_dict[key] == 2:
                    twoset.add(key)
                    match_value = 2
            if match_value == 2:
                chord_classification = str(twoset)

        # wenn nichtmal 2 gematcht werden können, ist viel schief gegangen und keine Klassifikation kann vorgenommen werden
        if chord_classification == 'no chord classification available yet':
            chord_classification = 'Die Akkordklassifikation ist gescheitert:('
            
        return match_value, chord_classification
        
    
    
    
    
    
    ### Das Ergebnis der Chord-Klassifikation ausgeben
    
    match_value, chord_classification = classify_detected_chord(detected_chord)
    
    if match_value != 2:
        print(f'\nDas Ergebnis der Akkord-Klassifizierung ist: {chord_classification}')
    detected_chord_inNotes = chord_numbers_to_notes(detected_chord)
    if match_value != 2:
        print(f'Diese 3 Töne wurden identifiziert: {detected_chord_inNotes}')
    # zusatz für 2er matching
    if match_value == 2:
        if description == 1:
            print(f'Davon stimmen {match_value} Töne mit dem Akkordmuster von {chord_classification} überein.')
            print('Es wird ein weiterer Test auf das Vorhandensein einer Quinte durchgeführt, um evtl ein konkreteres Ergenbis zu erhalten...')
        localize_q_and_put3(detected_chord, round=2)
        # Akkordklassifikation nochmal durchführen
        if description == 1:
            print('Neue Akkordklassifikation aufgrund des Quinttests:')
        match_value, chord_classification = classify_detected_chord(detected_chord)
        print(f'\nDas Ergebnis der Akkord-Klassifizierung ist: {chord_classification}')
        detected_chord_inNotes = chord_numbers_to_notes(detected_chord)
        print(f'Diese 3 Töne wurden identifiziert: {detected_chord_inNotes}')


# # Audios mit Klavierakkorden

# In[2]:


akkorderkennung('Audios/ADur.wav', description=1, percfilter=0)


# In[3]:


akkorderkennung('Audios/ADur_arp.wav', description=1, percfilter=0)


# # Audios mit Gitarrenakkorden

# In[4]:


akkorderkennung('Audios/am_strummed.wav', description=1, percfilter=0)


# In[5]:


akkorderkennung('Audios/c#m_strummed.wav', description=1, percfilter=0)


# In[6]:


akkorderkennung('Audios/dm_strummed.wav')


# In[7]:


akkorderkennung('Audios/am_arp_git.wav')


# In[8]:


akkorderkennung('Audios/c#m_arp_git.wav')


# In[9]:


akkorderkennung('Audios/dm_arp_git.wav', description=0)


# In[10]:


akkorderkennung('Audios/gm_arp_stopped_git.wav')


# In[11]:


akkorderkennung('Audios/em_arp_stopped_git.wav', description=0, percfilter=1)


# In[12]:


akkorderkennung('Audios/em_strummed_several_times.wav', description=0, percfilter=1)


# ### Anwendung der Akkorderkennung auf Sinuston Akkorde

# In[13]:


# Bauen einer Funktion, welche Sinustonakkorde generiert und mithilfe dieser dann Wav-Dateien erstellen, 
# die zur Akkordanalyse benutzt werden können 

import soundfile as sf


# Funktion zur Sinusakkorderstellung 
def generate_sinechord(f1, f2, f3, dur, ampl, sr):
    ''''f1 = Grundtonfrequenz, f2 = Terzfrequenz, f3 = Quintfrequenz, dur= duration, ampl= amplitude, sr= sampling rate'''
   
    t = np.linspace(0, dur, dur*sr, endpoint=False)
    
    f1sine = ampl * np.sin(2* np.pi*f1*t)
    f2sine = ampl * np.sin(2* np.pi*f2*t)
    f3sine = ampl * np.sin(2* np.pi*f3*t)
    
    # hört sich schrecklich an: sinechord = f1sine + f2sine + f3sine
    sinechord = np.concatenate([f1sine, f2sine, f3sine])
    #Abspielen zum Test
    #sd.play(f1sine, sr, blocking=True)
    #sd.play(f2sine, sr, blocking=True)
    #sd.play(f3sine, sr, blocking=True)
    sd.play(sinechord, sr, blocking=True)
    return sinechord


#ein paar Audiodateien zur Verwendung in der Akkorderkennungsanalyse erstellen
amoll = generate_sinechord(440, 528, 660, 1, 1, 44100)
sf.write('amoll-sinus.wav', amoll, 44100)

cisdur = generate_sinechord(277.18, 349.23, 415.3, 1, 1, 44100)
sf.write('cisdur-sinus.wav', cisdur, 44100)

bbdur = generate_sinechord(466.16, 587.33, 698.46, 1, 1, 44100)
sf.write('bbdur-sinus.wav', bbdur, 44100)

cmoll = generate_sinechord(1046.5, 1244.51, 1567.98, 1, 1, 44100)
sf.write('cmoll-sinus.wav', cmoll, 44100)


# #### Die Sinusakkorde werden alle korrekt klassifiziert, da hier keine störenden Nebeneffekte in der Audiodatei enthalten sind und somit die Chromawerte, wie in dem Plot zu erkennen ist, sehr sauber abgebildet werden.

# In[14]:


akkorderkennung('amoll-sinus.wav', description=0, percfilter=1)


# In[15]:


akkorderkennung('cisdur-sinus.wav', description=1, percfilter=1)


# In[16]:


akkorderkennung('bbdur-sinus.wav', description=1, percfilter=0)


# In[17]:


akkorderkennung('cmoll-sinus.wav', description=0, percfilter=0)


# ### Bemerkungen
# 
# Chroma_values ist ein 2 dimensionales array, welches die Stärke der Frequenzen der 12 Grundtöne abbildet.
# Die 12 Reihen sind also die unterschiedlichen Tonhöhen. Dabei gilt folgende Zuordnung:
# 
# 0=C
# 1=C#
# 2=D
# 3=D#
# 4=E
# 5=F
# 6=F#
# 7=G
# 8=G#
# 9=A
# 10=Bb
# 11=B

# ### Aufgetretene Probleme
# 
# - Fall: nur ein einzelner Ton ist bei nicht arpeggiertem Akkord gut in den Chroma_Klassen abgebildet
# - Rauschen außer bei Sinustondateien stört und erzeugt viel noise auf den chroma_values. Rauschfilterungsversuch ist gescheitert
#   Man könnte noch händisch die Sekunde vor und nach Tonerklingung abschneiden, um sauberere Dateien zur Analyse vorliegen zu       haben.
# - Umliegende Töne von tatsächlich gespielten bekommen auch oft gut Farbe in den Chroma values. Werden durch Funktionen, die  die 
#   benachbarten Töne erkennen, rausgeworfen und so versucht dieses Problem zu umgehen. Auch wird die Quinte, wenn vorhanden und     erkannt, genutzt um die Klassifikation zu schärfen 
# - Dinge, die Programm nicht kann bzw. nicht darauf ausgelegt ist: mehr als einen Akkord aus einer Datei auslesen. Unsinnsinput     abfangen. Andere Akkorde als 24 Grundakkorde erkennen.
# 
