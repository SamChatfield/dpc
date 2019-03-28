-module(tarry).
-export([start/0, node_main/2]).

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
    io:format("Nodes: ~p~n", [Nodes]),

    % Inform the node processes about the others that they are connected to
    send_connections(Topology, Nodes),

    % Do Tarry
    {_, InitiatorPid} = lists:keyfind(Initiator, 1, Nodes),
    Token = do_tarry(InitiatorPid),
    io:format("MAIN Token: ~p~n", [Token]),

    io:format("~nMAIN DONE~n").

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
    Pid = spawn(tarry, node_main, [self(), NodeName]),
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
    io:format("Main starting Tarry with initiator: ~p~n", [InitiatorPid]),
    InitiatorPid ! {tarry, self(), []},
    receive
        {tarry, Sender, Token} ->
            io:format("Main got Tarry done~n"),
            lists:reverse(Token)
    end.

%%
%% NODE PROCESS
%%

node_main(MainPid, NodeName) ->
    io:format("Spawned node ~p (~p) pointing back to main (~p)~n", [NodeName, self(), MainPid]),
    node_loop(MainPid, NodeName, []).

node_loop(MainPid, NodeName, ConnectedNodes) ->
    io:format("Node ~p (~p) connected to: ~p~n", [NodeName, self(), ConnectedNodes]),
    receive
        {connections, NodeConnections} ->
            io:format("Node ~p (~p) connected to: ~p~n", [NodeName, self(), NodeConnections]),
            MainPid ! {ack},
            node_tarry_loop(NodeName, NodeConnections, none, none, none),
            io:format("DONE~n")
    end.

node_tarry_loop(NodeName, _, Parent, [], Token) ->
    io:format("Node ~p with Parent=~p waiting for done~n", [NodeName, Parent]),
    receive
        {tarry, Sender, ReceivedToken} ->
            NewToken = [NodeName|ReceivedToken],
            io:format("Node ~p (~p) sending Token ~p back to parent ~p~n", [NodeName, self(), NewToken, Parent]),
            Parent ! {tarry, self(), NewToken}
    end;

node_tarry_loop(NodeName, NodeConnections, Parent, OldQueue, Token) ->
    io:format("Node ~p with Parent=~p, OldQueue=~p, Token=~p~n", [NodeName, Parent, OldQueue, Token]),
    receive
        {tarry, Sender, ReceivedToken} when Parent =:= none ->
            io:format("Node ~p (~p) received Token ~p from ~p for the first time~n", [NodeName, self(), ReceivedToken, Sender]),
            NewToken = [NodeName|ReceivedToken],
            ChildQueue = lists:keydelete(Sender, 2, NodeConnections),
            case ChildQueue of
                [] ->
                    io:format("Node ~p sending NewToken ~p back to sender ~p~n", [NodeName, NewToken, Sender]),
                    % Sender ! {done, NewToken};
                    Sender ! {tarry, self(), NewToken};
                [{ChildName, ChildPid}|NewQueue] ->
                    io:format("Node ~p sending NewToken ~p to child ~p~n", [NodeName, NewToken, ChildName]),
                    ChildPid ! {tarry, self(), NewToken},
                    node_tarry_loop(NodeName, NodeConnections, Sender, NewQueue, NewToken)
            end;
        {tarry, Sender, ReceivedToken} ->
            io:format("Node ~p (~p) received Token ~p from ~p~n", [NodeName, self(), ReceivedToken, Sender]),
            NewToken = [NodeName|ReceivedToken],
            [{ChildName, ChildPid}|NewQueue] = OldQueue,
            io:format("Node ~p sending NewToken ~p to child ~p~n", [NodeName, NewToken, ChildName]),
            ChildPid ! {tarry, self(), NewToken},
            node_tarry_loop(NodeName, NodeConnections, Parent, NewQueue, NewToken);
        X ->
            io:format("REC SOMETHING ELSE: ~p~n", [X])
    end,
    io:format("Node ~p (~p) stopped waiting~n", [NodeName, self()]).
