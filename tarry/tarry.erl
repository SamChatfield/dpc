-module(tarry).
-export([start/0]).

%%
%% MAIN PROCESS
%%

start() ->
    % Read the initiator and topology data from stdin
    {Initiator, Topology} = read_input(),
    io:format("Initiator: ~p~n", [Initiator]),
    io:format("Topology: ~p~n", [Topology]),

    % Spawn the nodes required
    Nodes = spawn_nodes(Topology),

    % Inform the node processes about the others that they are connected to
    % Uses acks to ensure that all connections are set up before the algorithm begins
    send_connections(Topology, Nodes),

    % Get the Pid of the Initiator
    {_, InitiatorPid} = lists:keyfind(Initiator, 1, Nodes),

    % Perform Tarry algorithm
    Token = do_tarry(InitiatorPid),

    % Print the solution to stdout
    io:format("Tarry Solution:~n~s~n", [string:join(Token, " ")]).

read_input() ->
    % Read the initiator as the only thing in the first line
    [Initiator|_] = read_input_line(),
    % Read the network topology from the remaining lines
    Topology = read_topology([]),
    {Initiator, Topology}.

read_input_line() ->
    case io:get_line("") of
        eof ->
            % End of the file
            eof;
        Data ->
            % Trim the trailing newline and split the line about the spaces
            string:tokens(string:strip(Data, right, $\n), " ")
    end.

read_topology(Topology) ->
    case read_input_line() of
        eof ->
            % End of the file, return the network Topology accumulated
            Topology;
        [Node|ConnectedNodes] ->
            % Add the line to the Topology accumulator and recurse
            read_topology([{Node, ConnectedNodes}|Topology])
    end.

spawn_nodes(Topology) ->
    [spawn_node(NodeName) || {NodeName, _} <- Topology].

spawn_node(NodeName) ->
    Pid = spawn(tarry_node, main, [self(), NodeName]),
    {NodeName, Pid}.

send_connections([], _) ->
    ok;

send_connections([{NodeName, ConnectedNodeNames}|Topology], Nodes) ->
    {_, NodePid} = lists:keyfind(NodeName, 1, Nodes),
    ConnectedNodes = [lists:keyfind(N, 1, Nodes) || N <- ConnectedNodeNames],
    NodePid ! {connections, ConnectedNodes},
    receive
        {ack} ->
            send_connections(Topology, Nodes)
    end.

do_tarry(InitiatorPid) ->
    InitiatorPid ! {tarry, self(), []},
    receive
        {tarry, _, Token} ->
            lists:reverse(Token)
    end.
