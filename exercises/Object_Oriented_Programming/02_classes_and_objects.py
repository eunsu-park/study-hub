"""
Exercise 02: Classes and Objects
Topic: Object-Oriented Programming

Practice class attributes, instance attributes, class methods, and static methods.
"""


class Temperature:
    """A temperature class with unit conversion.

    Class Attributes:
        ABSOLUTE_ZERO_C (float): -273.15

    Instance Attributes:
        celsius (float): Temperature in Celsius.

    Properties (read-only):
        fahrenheit: celsius * 9/5 + 32
        kelvin: celsius + 273.15

    Class Methods:
        from_fahrenheit(f): Create from Fahrenheit value.
        from_kelvin(k): Create from Kelvin value.

    Static Methods:
        is_freezing(celsius): Return True if celsius <= 0.

    Other:
        __repr__(): Return "Temperature(XX.XC)"
    """

    ABSOLUTE_ZERO_C = -273.15

    # TODO: Implement this class
    pass


class Playlist:
    """A music playlist.

    Class Attributes:
        total_playlists (int): Count of all playlists created (starts 0).

    Instance Attributes:
        name (str): Playlist name.
        songs (list): List of song title strings.

    Methods:
        add_song(title): Add a song. Raise ValueError if duplicate.
        remove_song(title): Remove a song. Raise ValueError if not found.
        shuffle(): Return a new list with songs in random order (don't modify original).
        __len__(): Return number of songs.
        __contains__(title): Return True if song is in playlist.
        __repr__(): Return "Playlist('name', N songs)"
    """

    total_playlists = 0

    # TODO: Implement this class
    pass


if __name__ == "__main__":
    # Test Temperature
    t1 = Temperature(100)
    assert t1.fahrenheit == 212.0
    assert t1.kelvin == 373.15

    t2 = Temperature.from_fahrenheit(32)
    assert t2.celsius == 0.0

    t3 = Temperature.from_kelvin(0)
    assert abs(t3.celsius - (-273.15)) < 0.01

    assert Temperature.is_freezing(0) is True
    assert Temperature.is_freezing(1) is False

    try:
        Temperature(-300)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    print(f"t1 = {t1}")
    print(f"t2 = {t2}")

    # Test Playlist
    p = Playlist("Road Trip")
    p.add_song("Bohemian Rhapsody")
    p.add_song("Hotel California")
    p.add_song("Stairway to Heaven")

    assert len(p) == 3
    assert "Hotel California" in p
    assert "Unknown Song" not in p

    p.remove_song("Hotel California")
    assert len(p) == 2

    try:
        p.add_song("Bohemian Rhapsody")
        assert False, "Should raise ValueError for duplicate"
    except ValueError:
        pass

    shuffled = p.shuffle()
    assert len(shuffled) == 2

    print(f"Playlist: {p}")
    print(f"Total playlists: {Playlist.total_playlists}")

    print("\nAll tests passed!")
